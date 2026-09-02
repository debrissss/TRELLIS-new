import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from PIL import Image
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from .. import models


class SparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt

    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'

        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []

        # Build each representation
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)

        return torch.stack(images)


class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)

        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
        stats['With sparse structure latents'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root, 'ss_latents', self.latent_model, f'{instance}.npz'))
        z = torch.tensor(latent['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
        }
        return pack


class TextConditionedSparseStructureLatent(TextConditionedMixin, SparseStructureLatent):
    """
    Text-conditioned sparse structure dataset
    """
    pass


class ImageConditionedSparseStructureLatent(ImageConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """
    pass


# ControlNet 改动：继承原 latent 数据集，保留 x_0 的读取方式，
# 只额外构造与该样本一一对应的原始三维 occupancy 条件。
class SparseStructureLatent_ControlNet(SparseStructureLatent):
    """
    Sparse-structure latent dataset with an aligned raw 3D ControlNet condition.

    ``x_0`` is loaded from the precomputed SS latent exactly as in
    :class:`SparseStructureLatent`. ``control`` is the corresponding
    ``[1, control_resolution, control_resolution, control_resolution]``
    occupancy grid, i.e. the same input representation used to train the
    Sparse Structure Encoder.
    """

    def __init__(
        self,
        roots: str,
        *,
        control_resolution: int = 64,
        **kwargs,
    ):
        self.control_resolution = control_resolution
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        # 原 flow 训练只要求预编码 latent；ControlNet 还要求原始 voxel 数据。
        metadata = metadata[metadata["voxelized"]]
        stats["Control voxels available"] = len(metadata)
        return metadata, stats

    def validate_metadata_files(self, root, metadata):
        metadata, stats = super().validate_metadata_files(root, metadata)
        # 在启动训练前过滤缺少 PLY 的样本，避免 DataLoader 运行中途失败。
        has_control_voxels = metadata["sha256"].apply(
            lambda sha256: os.path.isfile(
                os.path.join(root, "voxels", f"{sha256}.ply")
            )
        )
        metadata = metadata[has_control_voxels]
        stats["Control voxel files present"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        # 先复用原实现读取并归一化 x_0，确保 flow 训练目标没有变化。
        pack = super().get_instance(root, instance)
        position = utils3d.io.read_ply(
            os.path.join(root, "voxels", f"{instance}.ply")
        )[0]
        # ControlNet 改动：将 [-0.5, 0.5] 空间中的 PLY 点恢复为与
        # SS Encoder 训练输入一致的 [1, 64, 64, 64] 二值 occupancy。
        # 与 TRELLIS 原生 SparseStructure 和 encode_ss_latent.py 保持完全一致：
        # 不裁剪越界坐标，由原始索引行为直接暴露非法 voxel 数据。
        coords = (
            (torch.tensor(position) + 0.5) * self.control_resolution
        ).int().contiguous()
        control = torch.zeros(
            1,
            self.control_resolution,
            self.control_resolution,
            self.control_resolution,
            dtype=torch.float32,
        )
        control[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        # FlowMatchingTrainer 会把 x_0/cond 以外的字段作为 **kwargs
        # 原样透传给 denoiser，因此无需修改原 loss 公式。
        pack["control"] = control
        return pack


class TextConditionedSparseStructureLatent_ControlNet(
    TextConditionedMixin,
    SparseStructureLatent_ControlNet,
):
    """Text-conditioned SS Flow dataset with a raw 3D control grid."""

    pass


class ImageConditionedSparseStructureLatent_ControlNet(
    ImageConditionedMixin,
    SparseStructureLatent_ControlNet,
):
    """同时返回 DINO 图像条件和原始三维 occupancy 条件的 SS Flow 数据集。"""

    pass


# ControlNet 改动：FaceScan 的 paired 数据使用独立 control_voxels 和
# target latent 目录。该类不改变通用 TRELLIS/Facescape 数据集契约，避免
# 其他实验把完整监督 occupancy 当作三维条件。
class FaceScanSparseStructureLatent_ControlNet(SparseStructureLatent):
    """SS latent target paired with a normalized FaceScan control occupancy."""

    def __init__(
        self,
        roots: str,
        *,
        control_resolution: int = 64,
        control_voxel_dir: str = "control_voxels",
        **kwargs,
    ):
        self.control_resolution = control_resolution
        self.control_voxel_dir = control_voxel_dir
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata["control_voxelized"]]
        stats["Control voxels available"] = len(metadata)
        return metadata, stats

    def validate_metadata_files(self, root, metadata):
        metadata, stats = super().validate_metadata_files(root, metadata)
        latent_root = os.path.join(root, "ss_latents", self.latent_model)
        has_target_latent = metadata["sha256"].apply(
            lambda instance: os.path.isfile(
                os.path.join(latent_root, f"{instance}.npz")
            )
        )
        metadata = metadata[has_target_latent]
        stats["Target latent files present"] = len(metadata)

        has_control_voxel = metadata["sha256"].apply(
            lambda instance: os.path.isfile(
                os.path.join(root, self.control_voxel_dir, f"{instance}.ply")
            )
        )
        metadata = metadata[has_control_voxel]
        stats["Control voxel files present"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        # SparseStructureLatent 只读取由完整 target mesh 编码出的 x_0。
        pack = super().get_instance(root, instance)
        position = utils3d.io.read_ply(
            os.path.join(root, self.control_voxel_dir, f"{instance}.ply")
        )[0]
        coords = (
            (torch.tensor(position) + 0.5) * self.control_resolution
        ).int().contiguous()
        control = torch.zeros(
            1,
            self.control_resolution,
            self.control_resolution,
            self.control_resolution,
            dtype=torch.float32,
        )
        control[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        pack["control"] = control
        return pack


class FaceScanImageConditionedMixin_ControlNet:
    """Load the deterministic FaceScan normal-map image condition."""

    def __init__(
        self,
        roots,
        *,
        image_size: int = 518,
        image_filename: str = "up_normal.png",
        **kwargs,
    ):
        self.image_size = image_size
        self.image_filename = image_filename
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata["cond_rendered"]]
        stats["Normal-map conditions available"] = len(metadata)
        return metadata, stats

    def validate_metadata_files(self, root, metadata):
        metadata, stats = super().validate_metadata_files(root, metadata)
        has_image = metadata["sha256"].apply(
            lambda instance: os.path.isfile(
                os.path.join(
                    root,
                    "renders_cond",
                    # FaceScan 文件夹 ID 仅由数字组成，pandas 默认会把
                    # metadata.csv 的 sha256 列推断为整数。os.path.join
                    # 不接受整数，因此专用入口在构造路径时显式恢复字符串。
                    str(instance),
                    self.image_filename,
                )
            )
        )
        metadata = metadata[has_image]
        stats["Normal-map image files present"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        # 与 validate_metadata_files 保持一致，避免 DataLoader 读取真实样本时
        # 再次把 pandas/numpy 整数 ID 传给 os.path.join。
        instance = str(instance)
        image_path = os.path.join(
            root,
            "renders_cond",
            instance,
            self.image_filename,
        )
        # FaceScan up_normal.png 是 RGB 黑背景法线图，不具备 TRELLIS render
        # 的 alpha 通道；因此不套用 ImageConditionedMixin 的 alpha bbox 逻辑。
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.LANCZOS,
        )
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        pack["cond"] = image
        return pack


class ImageConditionedFaceScanSparseStructureLatent_ControlNet(
    FaceScanImageConditionedMixin_ControlNet,
    FaceScanSparseStructureLatent_ControlNet,
):
    """FaceScan normal-map + 3D occupancy conditioned SS Flow dataset."""

    pass
