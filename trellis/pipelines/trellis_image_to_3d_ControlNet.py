from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..models.sparse_structure_flow_ControlNet import (
    SparseStructureFlowModel_ControlNet,
)


# ControlNet 改动：复制原 image-to-3D pipeline，仅扩展 sparse structure
# 阶段；后续 structured latent 采样和解码流程保持原样。
class TrellisImageTo3DPipeline_ControlNet(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline_ControlNet":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(
            TrellisImageTo3DPipeline_ControlNet,
            TrellisImageTo3DPipeline_ControlNet,
        ).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline_ControlNet()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        # ControlNet 修复：父类只按 pipeline.json 加载模型，不能仅凭 pipeline
        # 类名推断内部 flow 已带控制分支，因此在初始化采样器前立即校验。
        new_pipeline._validate_controlnet_flow_model()

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline

    def _validate_controlnet_flow_model(self) -> SparseStructureFlowModel_ControlNet:
        """Return the SS ControlNet model or fail before sampling starts."""
        flow_model = self.models.get("sparse_structure_flow_model")
        if not isinstance(flow_model, SparseStructureFlowModel_ControlNet):
            actual = type(flow_model).__name__ if flow_model is not None else "None"
            raise TypeError(
                "TrellisImageTo3DPipeline_ControlNet requires "
                "models['sparse_structure_flow_model'] to be "
                "SparseStructureFlowModel_ControlNet, "
                f"but got {actual}. Use the dedicated ControlNet pipeline.json."
            )
        return flow_model

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        # dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        import os
        hub_dir = os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main')
        dinov2_model = torch.hub.load(hub_dir, name, pretrained=True, source='local')
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        control: Optional[Union[torch.Tensor, np.ndarray]] = None,
        prepared_control: Optional[torch.Tensor] = None,
        control_scale: Union[float, Sequence[float]] = 1.0,
        control_schedule: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            control (torch.Tensor or np.ndarray): Raw occupancy in
                [C,R,R,R] or [B,C,R,R,R]. The pipeline adds a batch dimension
                when needed and moves/casts it to the Control Encoder input.
            prepared_control (torch.Tensor): Strict precomputed tokens returned
                by prepare_control(); mutually exclusive with raw control.
            control_scale (float or sequence): Base residual strength shared by
                all Flow timesteps, or one base strength per control block.
            control_schedule (mapping, optional): Timestep gate applied to the
                base control scale. ``name='smoothstep'`` keeps full strength at
                ``t >= full_strength_t``, smoothly decays in the middle, and
                keeps ``min_scale`` at ``t <= min_strength_t``.
        """
        # Sample occupancy latent
        # 再次校验手工构造的 pipeline，避免绕过 from_pretrained() 后把
        # control 参数传入普通 SparseStructureFlowModel。
        flow_model = self._validate_controlnet_flow_model()
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if control is not None and prepared_control is not None:
            raise ValueError(
                "control and prepared_control are mutually exclusive"
            )
        if control_schedule is not None and (
            control is None and prepared_control is None
        ):
            raise ValueError(
                "control_schedule requires control or prepared_control"
            )

        # ControlNet 专用 pipeline 在进入 Euler sampler 前完成一次 64^3
        # encoder+projection；之后所有 step 及 CFG 正负分支复用同一 tensor。
        control_args = {}
        if control is not None:
            # Public pipeline 负责把用户友好的 occupancy 输入规范化；模型层的
            # prepare_control()/forward() 仍保持严格校验，便于尽早发现内部调用错误。
            # 这保留了改造前对 NumPy、CPU tensor 和 [C,R,R,R] 单样本的支持。
            if not isinstance(control, torch.Tensor):
                control = torch.as_tensor(control)
            if control.ndim == 4:
                control = control.unsqueeze(0)
            if control.ndim != 5:
                raise ValueError(
                    "control must have shape [C, R, R, R] or "
                    "[B, C, R, R, R]"
                )
            control = control.to(
                device=noise.device,
                dtype=flow_model.control_encoder.input_layer.weight.dtype,
            )
            flow_model._validate_raw_control(
                control, batch_size=num_samples, device=noise.device
            )
            # pipeline 推理不需要为可训练 projection 保留计算图；训练 trainer
            # 仍直接走模型的 raw control 路径并保留该层梯度。
            with torch.no_grad():
                prepared_control = flow_model.prepare_control(
                    control, batch_size=num_samples
                )
        if prepared_control is not None:
            flow_model.validate_prepared_control(
                prepared_control,
                batch_size=num_samples,
                device=noise.device,
            )
            control_args = {
                "prepared_control": prepared_control,
                "control_scale": control_scale,
            }
            if control_schedule is not None:
                control_args["control_schedule"] = control_schedule

        # Euler sampler 已支持 **kwargs 透传；三维条件在 CFG 的正、负图像分支
        # 中保持相同，因此图像 CFG 不会额外放大 control 强度。
        sampler_output = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **control_args,
            **sampler_params,
            verbose=True
        )
        self.last_sparse_structure_control_trace = list(
            getattr(sampler_output, "control_schedule_trace", [])
        )
        z_s = sampler_output.samples

        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        control: Optional[Union[torch.Tensor, np.ndarray]] = None,
        prepared_control: Optional[torch.Tensor] = None,
        control_scale: Union[float, Sequence[float]] = 1.0,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        control_schedule: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            control (torch.Tensor or np.ndarray): Raw 3D occupancy condition;
                accepts [C,R,R,R] or [B,C,R,R,R] on CPU or pipeline device.
            prepared_control (torch.Tensor): Tokens returned by the flow model's
                prepare_control(); mutually exclusive with raw control.
            control_scale (float or sequence): Control residual strength.
            control_schedule (mapping, optional): Smooth timestep gate for the
                ControlNet residual strength during SS Flow inference.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        # ControlNet 只约束 occupancy/稀疏结构生成；得到 coords 后仍调用
        # 原来的 SLat flow 和 decoder，避免扩大改动范围。
        coords = self.sample_sparse_structure(
            cond,
            num_samples,
            sparse_structure_sampler_params,
            control=control,
            prepared_control=prepared_control,
            control_scale=control_scale,
            control_schedule=control_schedule,
        )
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.

        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        control: Optional[Union[torch.Tensor, np.ndarray]] = None,
        prepared_control: Optional[torch.Tensor] = None,
        control_scale: Union[float, Sequence[float]] = 1.0,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
        control_schedule: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            control (torch.Tensor or np.ndarray): Raw 3D occupancy condition;
                accepts [C,R,R,R] or [B,C,R,R,R] on CPU or pipeline device.
            prepared_control (torch.Tensor): Strict precomputed tokens returned
                by the flow model; mutually exclusive with raw control.
            control_scale (float or sequence): Base ControlNet residual strength.
            control_schedule (mapping, optional): Smooth timestep gate for the
                ControlNet residual strength during SS Flow inference.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            # 多视图模式同样只在 SS Flow 阶段加入同一个三维条件。
            coords = self.sample_sparse_structure(
                cond,
                num_samples,
                sparse_structure_sampler_params,
                control=control,
                prepared_control=prepared_control,
                control_scale=control_scale,
                control_schedule=control_schedule,
            )
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
