from typing import *
import copy
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
        self.blocks.apply(convert_module_to_f16)
        self.control_blocks.apply(convert_module_to_f16)
        self.control_input_layer.apply(convert_module_to_f16)
        self.control_output_layers.apply(convert_module_to_f16)
        self.control_encoder.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
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
        for control_block, backbone_block in zip(
            self.control_blocks, self.blocks[:self.num_control_blocks]
        ):
            control_block.load_state_dict(backbone_block.state_dict())

    def _zero_control_projections(self) -> None:
        """Guarantee a zero ControlNet residual at initialization."""
        nn.init.constant_(self.control_input_layer.weight, 0)
        nn.init.constant_(self.control_input_layer.bias, 0)
        for layer in self.control_output_layers:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

    def _set_trainable_parameters(self) -> None:
        self.control_encoder.requires_grad_(False)
        if self.freeze_backbone:
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
        is_controlnet_checkpoint = any(
            key.startswith("control_") for key in state_dict.keys()
        )
        if is_controlnet_checkpoint:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        self._copy_backbone_to_control()
        self._zero_control_projections()
        self._set_trainable_parameters()
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        self.control_encoder.eval()
        if self.freeze_backbone:
            self.t_embedder.eval()
            self.input_layer.eval()
            self.blocks.eval()
            self.out_layer.eval()
            if self.share_mod:
                self.adaLN_modulation.eval()
        return self

    def _encode_control(self, control: torch.Tensor) -> torch.Tensor:
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

        with torch.no_grad():
            control_latent = self.control_encoder(
                control.float(), sample_posterior=False
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

        control_tokens = patchify(control_latent, self.patch_size)
        control_tokens = control_tokens.view(
            *control_tokens.shape[:2], -1
        ).permute(0, 2, 1).contiguous()
        control_tokens = control_tokens.type(self.dtype)
        return self.control_input_layer(control_tokens)

    def _get_control_scales(
        self,
        control_scale: Optional[Union[float, Sequence[float]]],
    ) -> List[float]:
        scale = self.control_scale if control_scale is None else control_scale
        if isinstance(scale, (float, int)):
            return [float(scale)] * self.num_control_blocks
        scales = list(scale)
        if len(scales) != self.num_control_blocks:
            raise ValueError(
                f"Expected {self.num_control_blocks} control scales, got {len(scales)}"
            )
        return scales

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        control: Optional[torch.Tensor] = None,
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

        control_h = None
        control_keep_mask = None
        control_scales = None
        if control is not None:
            if control.shape[0] == 1 and x.shape[0] > 1:
                control = control.repeat(x.shape[0], 1, 1, 1, 1)
            if control.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Control batch size {control.shape[0]} does not match input "
                    f"batch size {x.shape[0]}"
                )
            control = control.to(device=x.device)
            control_h = h + self._encode_control(control).type(self.dtype)
            control_scales = self._get_control_scales(control_scale)
            if self.training and self.control_dropout > 0:
                control_keep_mask = (
                    torch.rand(x.shape[0], 1, 1, device=x.device)
                    >= self.control_dropout
                ).type(self.dtype)

        for i, block in enumerate(self.blocks):
            h = block(h, t_emb, cond)
            if control_h is not None and i < self.num_control_blocks:
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
