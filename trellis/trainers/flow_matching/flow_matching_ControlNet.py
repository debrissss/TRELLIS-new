from typing import *
import copy
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers
from ...utils.general_utils import dict_reduce
from ...utils.dist_utils import read_file_dist
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    基于流匹配（Flow Matching）目标的扩散模型训练器。

    Args:
        models (dict[str, nn.Module]): 待训练的模型。
        dataset (torch.utils.data.Dataset): 数据集。
        output_dir (str): 输出目录。
        load_dir (str): 加载目录。
        step (int): 要加载的训练步数。
        batch_size (int): 批大小（Batch Size）。
        batch_size_per_gpu (int): 单个 GPU 的批大小。如果指定此参数，batch_size 将被忽略。
        batch_split (int): 通过梯度累积进行批次拆分的份数。
        max_steps (int): 最大训练步数。
        optimizer (dict): 优化器配置。
        lr_scheduler (dict): 学习率调度器配置。
        elastic (dict): 弹性显存管理配置。
        grad_clip (float or dict): 梯度裁剪配置。
        ema_rate (float or list): 指数移动平均（EMA）的衰减率。
        fp16_mode (str): FP16 精度模式。
            - None: 不使用 FP16。
            - 'inflat_all': 为所有参数保留一个放大的 fp32 主参数（master parameter）。
            - 'amp': 自动混合精度（Automatic Mixed Precision）。
        fp16_scale_growth (float): FP16 梯度反向传播的缩放增长率。
        finetune_ckpt (dict): 微调检查点（Checkpoint）配置。
        log_param_stats (bool): 是否记录参数统计信息。
        i_print (int): 打印日志的步数间隔。
        i_log (int): 记录日志的步数间隔。
        i_sample (int): 采样的步数间隔。
        i_save (int): 保存检查点的步数间隔。
        i_ddpcheck (int): DDP（分布式数据并行）检查的步数间隔。

        t_schedule (dict): 流匹配的时间步调度策略（Time schedule）。
        sigma_min (float): 最小噪声水平。
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

    # BEGIN detailed loss export: add weighted contribution fields for loss.txt.
    def export_loss_for_file(self, loss):
        export = super().export_loss_for_file(loss)
        if 'mse' in export:
            export['contribution'] = {'mse': export['mse']}
        return export
    # END detailed loss export.

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对给定的扩散步数进行数据扩散。
        换句话说，即从分布 q(x_t | x_0) 中进行采样。

        Args:
            x_0: 无噪声输入的张量，Shape 为 [N x C x ...]。
            t: 扩散步数（范围 [0-1]）的张量，Shape 为 [N]。
            noise: 如果指定此参数，将使用该指定噪声，而不是生成新的随机噪声。

        Returns:
            x_t: 在时间步 t 下，加噪后的 x_0 版本。
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        在时间步 t 下，从带噪的版本中恢复出原始图像。
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算时间步 t 下扩散过程的流速（Velocity）。
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        获取条件数据。
        """
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        获取推理时的条件数据。
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        获取扩散过程的采样器。
        """
        return samplers.FlowEulerSampler(self.sigma_min)

    def vis_cond(self, **kwargs):
        """
        可视化条件数据。
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        采样时间步。
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        计算单个时间步的训练损失。

        Args:
            x_0: 无噪声输入的张量，Shape 为 [N x C x ...]。
            cond: 额外条件数据的张量，Shape 为 [N x ...]。
            kwargs: 传给骨干网络（Backbone）的其他参数。

        Returns:
            一个字典，其中键为 "loss" 的值包含一个 Shape 为 [N] 的损失张量。
            也可能包含对应其他损失项的其他键。
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))

        return sample_dict


class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    基于流匹配目标和无分类器引导（Classifier-Free Guidance）的扩散模型训练器。

    Args:
        models (dict[str, nn.Module]): 待训练的模型。
        dataset (torch.utils.data.Dataset): 数据集。
        output_dir (str): 输出目录。
        load_dir (str): 加载目录。
        step (int): 要加载的训练步数。
        batch_size (int): 批大小（Batch Size）。
        batch_size_per_gpu (int): 单个 GPU 的批大小。如果指定此参数，batch_size 将被忽略。
        batch_split (int): 通过梯度累积进行批次拆分的份数。
        max_steps (int): 最大训练步数。
        optimizer (dict): 优化器配置。
        lr_scheduler (dict): 学习率调度器配置。
        elastic (dict): 弹性显存管理配置。
        grad_clip (float or dict): 梯度裁剪配置。
        ema_rate (float or list): 指数移动平均（EMA）的衰减率。
        fp16_mode (str): FP16 精度模式。
            - None: 不使用 FP16。
            - 'inflat_all': 为所有参数保留一个放大的 fp32 主参数（master parameter）。
            - 'amp': 自动混合精度（Automatic Mixed Precision）。
        fp16_scale_growth (float): FP16 梯度反向传播的缩放增长率。
        finetune_ckpt (dict): 微调检查点（Checkpoint）配置。
        log_param_stats (bool): 是否记录参数统计信息。
        i_print (int): 打印日志的步数间隔。
        i_log (int): 记录日志的步数间隔。
        i_sample (int): 采样的步数间隔。
        i_save (int): 保存检查点的步数间隔。
        i_ddpcheck (int): DDP（分布式数据并行）检查的步数间隔。

        t_schedule (dict): 流匹配的时间步调度策略（Time schedule）。
        sigma_min (float): 最小噪声水平。
        p_uncond (float): 丢弃条件数据（用于无分类器引导的无条件训练）的概率。
    """
    pass


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    基于流匹配目标和无分类器引导的文本条件扩散模型训练器。

    Args:
        models (dict[str, nn.Module]): 待训练的模型。
        dataset (torch.utils.data.Dataset): 数据集。
        output_dir (str): 输出目录。
        load_dir (str): 加载目录。
        step (int): 要加载的训练步数。
        batch_size (int): 批大小（Batch Size）。
        batch_size_per_gpu (int): 单个 GPU 的批大小。如果指定此参数，batch_size 将被忽略。
        batch_split (int): 通过梯度累积进行批次拆分的份数。
        max_steps (int): 最大训练步数。
        optimizer (dict): 优化器配置。
        lr_scheduler (dict): 学习率调度器配置。
        elastic (dict): 弹性显存管理配置。
        grad_clip (float or dict): 梯度裁剪配置。
        ema_rate (float or list): 指数移动平均（EMA）的衰减率。
        fp16_mode (str): FP16 精度模式。
            - None: 不使用 FP16。
            - 'inflat_all': 为所有参数保留一个放大的 fp32 主参数（master parameter）。
            - 'amp': 自动混合精度（Automatic Mixed Precision）。
        fp16_scale_growth (float): FP16 梯度反向传播的缩放增长率。
        finetune_ckpt (dict): 微调检查点（Checkpoint）配置。
        log_param_stats (bool): 是否记录参数统计信息。
        i_print (int): 打印日志的步数间隔。
        i_log (int): 记录日志的步数间隔。
        i_sample (int): 采样的步数间隔。
        i_save (int): 保存检查点的步数间隔。
        i_ddpcheck (int): DDP（分布式数据并行）检查的步数间隔。

        t_schedule (dict): 流匹配的时间步调度策略（Time schedule）。
        sigma_min (float): 最小噪声水平。
        p_uncond (float): 丢弃条件数据（用于无分类器引导的无条件训练）的概率。
        text_cond_model (str): 文本条件模型。
    """
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    基于流匹配目标和无分类器引导的图像条件扩散模型训练器。

    Args:
        models (dict[str, nn.Module]): 待训练的模型。
        dataset (torch.utils.data.Dataset): 数据集。
        output_dir (str): 输出目录。
        load_dir (str): 加载目录。
        step (int): 要加载的训练步数。
        batch_size (int): 批大小（Batch Size）。
        batch_size_per_gpu (int): 单个 GPU 的批大小。如果指定此参数，batch_size 将被忽略。
        batch_split (int): 通过梯度累积进行批次拆分的份数。
        max_steps (int): 最大训练步数。
        optimizer (dict): 优化器配置。
        lr_scheduler (dict): 学习率调度器配置。
        elastic (dict): 弹性显存管理配置。
        grad_clip (float or dict): 梯度裁剪配置。
        ema_rate (float or list): 指数移动平均（EMA）的衰减率。
        fp16_mode (str): FP16 精度模式。
            - None: 不使用 FP16。
            - 'inflat_all': 为所有参数保留一个放大的 fp32 主参数（master parameter）。
            - 'amp': 自动混合精度（Automatic Mixed Precision）。
        fp16_scale_growth (float): FP16 梯度反向传播的缩放增长率。
        finetune_ckpt (dict): 微调检查点（Checkpoint）配置。
        log_param_stats (bool): 是否记录参数统计信息。
        i_print (int): 打印日志的步数间隔。
        i_log (int): 记录日志的步数间隔.
        i_sample (int): 采样的步数间隔。
        i_save (int): 保存检查点的步数间隔。
        i_ddpcheck (int): DDP（分布式数据并行）检查的步数间隔。

        t_schedule (dict): 流匹配的时间步调度策略（Time schedule）。
        sigma_min (float): 最小噪声水平。
        p_uncond (float): 丢弃条件数据（用于无分类器引导的无条件训练）的概率。
        image_cond_model (str): 图像条件模型。
    """
    pass


# ControlNet 改动：原 FlowMatchingTrainer 的 diffuse、velocity target 和 MSE
# 全部保持不变；这里只处理“旧主干 checkpoint 缺少 control_* 参数”的初始化问题。
class ImageConditionedFlowMatchingCFGTrainer_ControlNet(
    ImageConditionedFlowMatchingCFGTrainer
):
    """
    Image-conditioned flow-matching trainer with base-to-ControlNet checkpoint
    initialization support.

    The ordinary trainer assumes the finetune checkpoint already contains
    every trainable parameter. A base SS Flow checkpoint has no ``control_*``
    tensors, so this override lets the ControlNet model expand that checkpoint
    first and then initializes optimizer master parameters from the complete
    model state.
    """

    def finetune_from(self, finetune_ckpt):
        if self.is_master:
            print("\nFinetuning ControlNet from:")
            for name, path in finetune_ckpt.items():
                print(f"  - {name}: {path}")

        model_ckpts = {}
        for name, model in self.models.items():
            if name in finetune_ckpt:
                # SparseStructureFlowModel_ControlNet.load_state_dict 会识别这是
                # 原 SS Flow 权重，并自动复制前 N 层到控制分支、保持零注入。
                base_ckpt = torch.load(
                    read_file_dist(finetune_ckpt[name]),
                    map_location=self.device,
                    weights_only=True,
                )
                model.load_state_dict(base_ckpt)
                if self.fp16_mode == "inflat_all":
                    model.convert_to_fp16()
            elif self.is_master:
                print(f"Warning: {name} not found in finetune_ckpt, skipped.")

            # ControlNet 改动：不能继续使用缺少 control_* key 的旧 checkpoint
            # 初始化 master params；必须改用模型扩展后的完整 state_dict。
            model_ckpts[name] = model.state_dict()

        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print("Done.")
        if self.world_size > 1:
            self.check_ddp()
