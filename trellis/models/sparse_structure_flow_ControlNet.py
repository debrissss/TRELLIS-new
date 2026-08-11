from typing import *
import copy
import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify
from .sparse_structure_vae import SparseStructureEncoder


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ControlNet 改动：保留原 SparseStructureFlowModel 的主干命名和计算顺序，
# 在同一个模型内增加独立的三维条件编码器、控制分支和零初始化注入层。
class SparseStructureFlowModel_ControlNet(nn.Module):
    """
    ControlNet variant of :class:`SparseStructureFlowModel`.

    The original SS Flow backbone is kept frozen. A trainable copy of its first
    ``num_control_blocks`` transformer blocks processes a spatially aligned 3D
    condition. Zero-initialized linear layers inject the control features into
    the frozen backbone, so loading a base SS Flow checkpoint preserves the
    original denoiser exactly before ControlNet training.

    The 3D condition follows the Sparse Structure Encoder training input:
    ``[B, control_channels, control_resolution, control_resolution,
    control_resolution]`` (normally a ``[B, 1, 64, 64, 64]`` occupancy grid).
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        # ControlNet 新增参数：描述原始三维条件、控制分支容量和训练策略。
        control_channels: int = 1,
        control_resolution: int = 64,
        control_encoder_args: Optional[dict] = None,
        control_encoder_ckpt: Optional[str] = None,
        control_latent_normalization: Optional[dict] = None,
        num_control_blocks: int = 8,
        control_dropout: float = 0.0,
        control_scale: float = 1.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if not 1 <= num_control_blocks <= num_blocks:
            raise ValueError(
                f"num_control_blocks must be in [1, {num_blocks}], got {num_control_blocks}"
            )
        if not 0.0 <= control_dropout <= 1.0:
            raise ValueError(f"control_dropout must be in [0, 1], got {control_dropout}")

        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.control_channels = control_channels
        self.control_resolution = control_resolution
        self.num_control_blocks = num_control_blocks
        self.control_dropout = control_dropout
        self.control_scale = control_scale
        self.freeze_backbone = freeze_backbone

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        # ControlNet 改动：条件输入不是二维 hint，而是 SS Encoder 训练时使用的
        # 64^3 occupancy。冻结的 SS Encoder 将其压缩到与 x_t 对齐的 16^3 latent。
        if control_encoder_args is None:
            control_encoder_args = {
                "in_channels": control_channels,
                "latent_channels": in_channels,
                "num_res_blocks": 2,
                "num_res_blocks_middle": 2,
                "channels": [32, 128, 512],
                "use_fp16": use_fp16,
            }
        else:
            control_encoder_args = dict(control_encoder_args)
            control_encoder_args.setdefault("in_channels", control_channels)
            control_encoder_args.setdefault("latent_channels", in_channels)
            control_encoder_args.setdefault("use_fp16", use_fp16)
        if control_encoder_args["in_channels"] != control_channels:
            raise ValueError(
                "control_encoder_args.in_channels must match control_channels"
            )
        if control_encoder_args["latent_channels"] != in_channels:
            raise ValueError(
                "The control encoder latent channels must match SS Flow in_channels"
            )

        self.control_encoder = SparseStructureEncoder(**control_encoder_args)

        # ControlNet 改动：
        # 1. control_input_layer 将编码后的 3D latent token 投影到主干 hidden size；
        # 2. control_blocks 深拷贝主干前 N 层，作为独立可训练分支；
        # 3. 每个 control_output_layer 把控制特征作为残差写回对应主干层。
        self.control_input_layer = nn.Linear(
            in_channels * patch_size**3, model_channels
        )
        self.control_blocks = nn.ModuleList([
            copy.deepcopy(self.blocks[i]) for i in range(num_control_blocks)
        ])
        self.control_output_layers = nn.ModuleList([
            nn.Linear(model_channels, model_channels)
            for _ in range(num_control_blocks)
        ])

        # 若 x_0 在训练时做过逐通道归一化，控制 latent 必须使用相同统计量，
        # 否则控制分支与 flow 主干看到的 latent 数值分布会不一致。
        if control_latent_normalization is not None:
            mean = torch.tensor(control_latent_normalization["mean"]).float()
            std = torch.tensor(control_latent_normalization["std"]).float()
            if mean.numel() != in_channels or std.numel() != in_channels:
                raise ValueError(
                    "control latent normalization must have one value per latent channel"
                )
            self.register_buffer(
                "control_latent_mean", mean.view(1, -1, 1, 1, 1)
            )
            self.register_buffer(
                "control_latent_std", std.view(1, -1, 1, 1, 1)
            )
        else:
            self.control_latent_mean = None
            self.control_latent_std = None

        self.initialize_weights()
        # 先复制主干权重，再将输入/输出投影清零；这样控制分支具有预训练能力，
        # 但初始化时注入主干的残差严格为 0，不改变原 SS Flow 的输出。
        self._copy_backbone_to_control()
        self._zero_control_projections()
        if control_encoder_ckpt is not None:
            self.load_control_encoder(control_encoder_ckpt)
        self._set_trainable_parameters()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        # ControlNet 修复：动态转换时必须同步外层状态；forward 和
        # _encode_control 都依据 self.dtype 选择计算精度。
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.control_blocks.apply(convert_module_to_f16)
        self.control_input_layer.apply(convert_module_to_f16)
        self.control_output_layers.apply(convert_module_to_f16)
        self.control_encoder.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.control_blocks.apply(convert_module_to_f32)
        self.control_input_layer.apply(convert_module_to_f32)
        self.control_output_layers.apply(convert_module_to_f32)
        self.control_encoder.convert_to_fp32()

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _copy_backbone_to_control(self) -> None:
        """Initialize the trainable branch from the corresponding base blocks."""
        # ControlNet 改动：对应层逐一复制，而不是随机初始化控制 Transformer。
        for control_block, backbone_block in zip(
            self.control_blocks, self.blocks[:self.num_control_blocks]
        ):
            control_block.load_state_dict(backbone_block.state_dict())

    def _zero_control_projections(self) -> None:
        """Guarantee a zero ControlNet residual at initialization."""
        # ControlNet 的关键安全设计：所有新增残差通路从恒等于 0 开始，
        # 因而未训练的 ControlNet 与原始预训练 SS Flow 完全等价。
        nn.init.constant_(self.control_input_layer.weight, 0)
        nn.init.constant_(self.control_input_layer.bias, 0)
        for layer in self.control_output_layers:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

    def _set_trainable_parameters(self) -> None:
        # SS Encoder 只负责提供稳定的三维 latent 表示，不参与 ControlNet 微调。
        self.control_encoder.requires_grad_(False)
        if self.freeze_backbone:
            # 冻结原 flow 的时间编码、输入层、24 个 block 和输出层；
            # 优化器最终只会收到 control_input/blocks/output 三部分参数。
            frozen_modules = [
                self.t_embedder,
                self.input_layer,
                self.blocks,
                self.out_layer,
            ]
            if self.share_mod:
                frozen_modules.append(self.adaLN_modulation)
            for module in frozen_modules:
                module.requires_grad_(False)
        self.control_input_layer.requires_grad_(True)
        self.control_blocks.requires_grad_(True)
        self.control_output_layers.requires_grad_(True)

    def load_control_encoder(self, checkpoint: str) -> None:
        """Load the frozen SS Encoder used to encode the raw 3D condition."""
        if checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.control_encoder.load_state_dict(state_dict)
        self.control_encoder.eval()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """
        Accept either a complete ControlNet checkpoint or a base SS Flow one.

        A base checkpoint has no ``control_*`` keys. In that case the matching
        backbone weights are loaded non-strictly and then copied into the
        trainable control blocks, while the zero projections and SS Encoder
        keep their constructor initialization.
        """
        # 通过 control_* key 区分两种输入：
        # - 完整 ControlNet checkpoint：严格恢复全部参数；
        # - 原 SS Flow checkpoint：非严格加载主干，再补建控制分支。
        is_controlnet_checkpoint = any(
            key.startswith("control_") for key in state_dict.keys()
        )
        if is_controlnet_checkpoint:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        # ControlNet 专用兼容边界：base SS Flow 只能缺少本类新增的
        # control_* 参数/缓冲区。过去直接返回 non-strict 结果会把损坏的主干
        # checkpoint（缺 backbone 或夹带未知 key）也当作合法 base 权重。
        invalid_missing = [
            key for key in result.missing_keys
            if not key.startswith("control_")
        ]
        if invalid_missing or result.unexpected_keys:
            details = []
            if invalid_missing:
                details.append(
                    "backbone missing keys: " + ", ".join(invalid_missing)
                )
            if result.unexpected_keys:
                details.append(
                    "unexpected keys: " + ", ".join(result.unexpected_keys)
                )
            raise RuntimeError(
                "Invalid base SparseStructureFlow checkpoint; "
                + "; ".join(details)
            )
        self._copy_backbone_to_control()
        self._zero_control_projections()
        self._set_trainable_parameters()
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        # 即使外部调用 model.train()，冻结模块仍保持 eval，避免条件编码器状态漂移。
        self.control_encoder.eval()
        if self.freeze_backbone:
            self.t_embedder.eval()
            self.input_layer.eval()
            self.blocks.eval()
            self.out_layer.eval()
            if self.share_mod:
                self.adaLN_modulation.eval()
        return self

    @property
    def control_token_count(self) -> int:
        return (self.resolution // self.patch_size) ** 3

    def _validate_raw_control(
        self,
        control: torch.Tensor,
        *,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if not isinstance(control, torch.Tensor):
            raise TypeError("control must be a torch.Tensor")
        if control.ndim != 5:
            raise ValueError(
                "control must have shape "
                "[B, control_channels, R, R, R]"
            )
        expected_shape = [
            control.shape[0],
            self.control_channels,
            *[self.control_resolution] * 3,
        ]
        if list(control.shape) != expected_shape:
            raise ValueError(
                f"Control shape mismatch, got {list(control.shape)}, "
                f"expected {expected_shape}"
            )
        if batch_size is not None and control.shape[0] not in (1, batch_size):
            raise ValueError(
                f"Control batch size must be 1 or {batch_size}, "
                f"got {control.shape[0]}"
            )
        expected_device = self.device if device is None else device
        if control.device != expected_device:
            raise ValueError(
                f"Control device mismatch, got {control.device}, "
                f"expected {expected_device}"
            )
        expected_dtype = self.control_encoder.input_layer.weight.dtype
        if control.dtype != expected_dtype:
            raise TypeError(
                f"Control dtype mismatch, got {control.dtype}, "
                f"expected {expected_dtype}"
            )

    def validate_prepared_control(
        self,
        prepared_control: torch.Tensor,
        *,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """严格校验已编码、已投影且可跨 Euler/CFG forward 复用的 token。"""
        if not isinstance(prepared_control, torch.Tensor):
            raise TypeError("prepared_control must be a torch.Tensor")
        if prepared_control.ndim != 3:
            raise ValueError(
                "prepared_control must have shape "
                "[B, control_token_count, model_channels]"
            )
        expected_shape = [
            prepared_control.shape[0],
            self.control_token_count,
            self.model_channels,
        ]
        if list(prepared_control.shape) != expected_shape:
            raise ValueError(
                f"Prepared control shape mismatch, got "
                f"{list(prepared_control.shape)}, expected {expected_shape}"
            )
        if (
            batch_size is not None
            and prepared_control.shape[0] not in (1, batch_size)
        ):
            raise ValueError(
                f"Prepared control batch size must be 1 or {batch_size}, "
                f"got {prepared_control.shape[0]}"
            )
        expected_device = self.device if device is None else device
        if prepared_control.device != expected_device:
            raise ValueError(
                f"Prepared control device mismatch, got "
                f"{prepared_control.device}, expected {expected_device}"
            )
        expected_dtype = self.control_input_layer.weight.dtype
        if prepared_control.dtype != expected_dtype:
            raise TypeError(
                f"Prepared control dtype mismatch, got "
                f"{prepared_control.dtype}, expected {expected_dtype}"
            )

    def prepare_control(
        self,
        control: torch.Tensor,
        *,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """将 raw occupancy 编码并投影一次，供整个采样轨迹复用。"""
        self._validate_raw_control(control, batch_size=batch_size)

        # 编码器被冻结，因此显式关闭 autograd，减少 64^3 Conv3D 的显存占用。
        with torch.no_grad():
            control_latent = self.control_encoder(
                control, sample_posterior=False
            )
        expected_latent_shape = [
            control.shape[0],
            self.in_channels,
            *[self.resolution] * 3,
        ]
        if list(control_latent.shape) != expected_latent_shape:
            raise ValueError(
                f"Encoded control shape mismatch, got {list(control_latent.shape)}, "
                f"expected {expected_latent_shape}. Check control_resolution and "
                "control_encoder_args."
            )
        if self.control_latent_mean is not None:
            control_latent = (
                control_latent - self.control_latent_mean
            ) / self.control_latent_std

        # 使用与 x_t 完全相同的 patchify 顺序，使每个 control token 与主干
        # 3D token 在空间位置上一一对应。
        control_tokens = patchify(control_latent, self.patch_size)
        control_tokens = control_tokens.view(
            *control_tokens.shape[:2], -1
        ).permute(0, 2, 1).contiguous()
        # 以投影层权重为最终 dtype 依据，避免外部动态精度转换后出现
        # mat1/mat2 dtype 不一致。
        control_tokens = control_tokens.to(
            dtype=self.control_input_layer.weight.dtype
        )
        prepared_control = self.control_input_layer(control_tokens)
        self.validate_prepared_control(
            prepared_control, batch_size=batch_size
        )
        return prepared_control

    def _get_control_scales(
        self,
        control_scale: Optional[Union[float, Sequence[float]]],
    ) -> List[float]:
        scale = self.control_scale if control_scale is None else control_scale

        # ControlNet 修复：先把标量、Tensor、NumPy 和普通序列统一成列表，
        # 再逐项检查类型和有限性，防止 NaN/Inf 污染整条采样轨迹。
        if isinstance(scale, torch.Tensor):
            if scale.ndim == 0:
                raw_scales = [scale]
            elif scale.ndim == 1:
                raw_scales = list(scale.unbind())
            else:
                raise ValueError(
                    "control_scale tensor must be scalar or one-dimensional"
                )
        elif isinstance(scale, np.ndarray):
            if scale.ndim == 0:
                raw_scales = [scale.item()]
            elif scale.ndim == 1:
                raw_scales = scale.tolist()
            else:
                raise ValueError(
                    "control_scale array must be scalar or one-dimensional"
                )
        elif isinstance(scale, numbers.Real) and not isinstance(scale, bool):
            raw_scales = [scale]
        elif isinstance(scale, (str, bytes)):
            raise TypeError("control_scale must be numeric, not a string")
        else:
            try:
                raw_scales = list(scale)
            except TypeError as exc:
                raise TypeError(
                    "control_scale must be a real scalar or a numeric sequence"
                ) from exc

        if len(raw_scales) == 1:
            raw_scales = raw_scales * self.num_control_blocks
        elif len(raw_scales) != self.num_control_blocks:
            raise ValueError(
                f"Expected 1 or {self.num_control_blocks} control scales, "
                f"got {len(raw_scales)}"
            )

        scales = []
        for index, value in enumerate(raw_scales):
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    raise TypeError(
                        f"control_scale[{index}] must be a scalar tensor"
                    )
                value = value.detach().item()
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(
                    f"control_scale[{index}] must be a real number, "
                    f"got {type(value).__name__}"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"control_scale[{index}] must be finite, got {value}"
                )
            scales.append(value)
        return scales

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        prepared_control: Optional[torch.Tensor] = None,
        control_scale: Optional[Union[float, Sequence[float]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)

        if control is not None and prepared_control is not None:
            raise ValueError(
                "control and prepared_control are mutually exclusive"
            )

        # 两种 control 都为空时完全跳过新增分支，可退化为原 SS Flow 推理。
        control_h = None
        control_keep_mask = None
        control_scales = None
        if control is not None:
            self._validate_raw_control(
                control, batch_size=x.shape[0], device=x.device
            )
            prepared_control = self.prepare_control(
                control, batch_size=x.shape[0]
            )
        if prepared_control is not None:
            self.validate_prepared_control(
                prepared_control, batch_size=x.shape[0], device=x.device
            )
            # 以主干 token 为起点叠加三维 hint，等价于
            # ControlNet-Transformer 的 x + zero_linear(condition) 初始化方式。
            # batch=1 的 token 由 PyTorch 广播到采样 batch，避免物理 repeat。
            control_h = h + prepared_control
            control_scales = self._get_control_scales(control_scale)
            if self.training and self.control_dropout > 0:
                # 可选的按样本随机 branch-drop 正则。它只让部分训练样本不更新
                # 控制残差，不能替代独立的无控制目标，也不应宣称可由此训练出
                # “无控制能力”；图像 CFG 的 p_uncond 语义与本参数相互独立。
                control_keep_mask = (
                    torch.rand(x.shape[0], 1, 1, device=x.device)
                    >= self.control_dropout
                ).type(self.dtype)

        for i, block in enumerate(self.blocks):
            h = block(h, t_emb, cond)
            if control_h is not None and i < self.num_control_blocks:
                # 控制 block 与对应主干 block 使用相同的时间和图像条件；
                # 经零初始化 Linear 后，残差写回 h，并继续流向后续冻结主干层。
                control_h = self.control_blocks[i](control_h, t_emb, cond)
                residual = self.control_output_layers[i](control_h)
                if control_keep_mask is not None:
                    residual = residual * control_keep_mask
                h = h + residual * control_scales[i]
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h
