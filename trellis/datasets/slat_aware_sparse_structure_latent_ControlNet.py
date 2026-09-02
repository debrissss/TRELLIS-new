import os
from typing import Optional

import numpy as np
import torch

from .sparse_structure_latent_ControlNet import (
    ImageConditionedFaceScanSparseStructureLatent_ControlNet,
)
from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices


class ImageConditionedFaceScanSLatAwareSparseStructureLatent_ControlNet(
    ImageConditionedFaceScanSparseStructureLatent_ControlNet,
):
    """FaceScan SS samples paired with frozen-teacher SLat targets.

    The SLat coordinates define a fixed candidate topology during distillation.
    This avoids pretending that the dynamic ``threshold + argwhere`` boundary is
    differentiable while still exposing the SS occupancy probabilities at every
    ground-truth SLat site to the frozen teacher.
    """

    def __init__(
        self,
        roots: str,
        *,
        slat_latent_model: str,
        slat_normalization: Optional[dict] = None,
        slat_resolution: int = 64,
        slat_feature_channels: int = 8,
        max_num_voxels: int = 32768,
        **kwargs,
    ):
        self.slat_latent_model = slat_latent_model
        self.slat_normalization = slat_normalization
        self.slat_resolution = int(slat_resolution)
        self.slat_feature_channels = int(slat_feature_channels)
        self.max_num_voxels = int(max_num_voxels)
        if self.slat_resolution <= 0:
            raise ValueError("slat_resolution must be positive")
        if self.slat_feature_channels <= 0:
            raise ValueError("slat_feature_channels must be positive")
        if self.max_num_voxels <= 0:
            raise ValueError("max_num_voxels must be positive")
        if slat_normalization is not None:
            if "mean" not in slat_normalization or "std" not in slat_normalization:
                raise ValueError("slat_normalization requires mean and std")
            self.slat_mean = torch.tensor(
                slat_normalization["mean"], dtype=torch.float32
            ).reshape(1, -1)
            self.slat_std = torch.tensor(
                slat_normalization["std"], dtype=torch.float32
            ).reshape(1, -1)
            expected_shape = (1, self.slat_feature_channels)
            if tuple(self.slat_mean.shape) != expected_shape:
                raise ValueError(
                    "SLat normalization mean length must match "
                    f"slat_feature_channels={self.slat_feature_channels}"
                )
            if tuple(self.slat_std.shape) != expected_shape:
                raise ValueError(
                    "SLat normalization std length must match "
                    f"slat_feature_channels={self.slat_feature_channels}"
                )
            if not torch.isfinite(self.slat_mean).all():
                raise ValueError("SLat normalization mean must be finite")
            if not torch.isfinite(self.slat_std).all() or torch.any(
                self.slat_std <= 0
            ):
                raise ValueError("SLat normalization std must be finite and positive")
        super().__init__(roots, **kwargs)

        self.loads = [
            int(self.metadata.loc[instance, "num_voxels"])
            for _, instance in self.instances
        ]

    @property
    def slat_metadata_column(self) -> str:
        return f"latent_{self.slat_latent_model}"

    @property
    def slat_voxel_count_column(self) -> str:
        return f"num_voxels_{self.slat_latent_model}"

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        column = self.slat_metadata_column
        if column not in metadata.columns:
            raise KeyError(
                f"Missing metadata column {column!r}. Generate paired SLat "
                "latents with dataset_toolkits/encode_latent.py and merge its "
                "completion records into metadata.csv before distillation."
            )
        # ``read_csv`` represents a partially populated boolean column as
        # object dtype (True/NaN). Comparing explicitly avoids pandas' pending
        # silent-downcast behavior while retaining only completed records.
        metadata = metadata[metadata[column].eq(True)]
        stats["Paired SLat latents"] = len(metadata)
        # Only a SLat-specific count can enforce the teacher sequence limit.
        # Legacy mesh voxel counts are not assumed to match encoded SLat sites.
        count_column = self.slat_voxel_count_column
        if count_column in metadata.columns:
            metadata = metadata[
                metadata[count_column].notna()
                & (metadata[count_column] <= self.max_num_voxels)
            ]
            stats[f"SLat voxels <= {self.max_num_voxels}"] = len(metadata)
        else:
            stats["SLat voxel count checked while loading"] = len(metadata)
        return metadata, stats

    def validate_metadata_files(self, root, metadata):
        metadata, stats = super().validate_metadata_files(root, metadata)
        latent_root = os.path.join(
            root, "latents", self.slat_latent_model
        )
        has_slat = metadata["sha256"].apply(
            lambda instance: os.path.isfile(
                os.path.join(latent_root, f"{instance}.npz")
            )
        )
        metadata = metadata[has_slat]
        stats["Paired SLat files present"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        latent_path = os.path.join(
            root,
            "latents",
            self.slat_latent_model,
            f"{instance}.npz",
        )
        with np.load(latent_path) as latent:
            raw_coords = torch.as_tensor(latent["coords"])
            feats = torch.tensor(latent["feats"], dtype=torch.float32)
        if raw_coords.is_floating_point() and not torch.equal(
            raw_coords, raw_coords.round()
        ):
            raise ValueError(f"Non-integral SLat coordinates in {latent_path}")
        coords = raw_coords.to(dtype=torch.int32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Invalid SLat coords shape {tuple(coords.shape)} in "
                f"{latent_path}; expected [N, 3]"
            )
        if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Invalid SLat feats shape {tuple(feats.shape)} in "
                f"{latent_path}; expected [N, C] aligned with coords"
            )
        if coords.shape[0] == 0:
            raise ValueError(f"Empty SLat topology in {latent_path}")
        if coords.shape[0] > self.max_num_voxels:
            raise ValueError(
                f"SLat topology in {latent_path} has {coords.shape[0]} sites; "
                f"limit is {self.max_num_voxels}"
            )
        if coords.min() < 0 or coords.max() >= self.slat_resolution:
            raise ValueError(
                f"SLat coordinates in {latent_path} must be within "
                f"[0, {self.slat_resolution - 1}]"
            )
        if torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
            raise ValueError(f"Duplicate SLat coordinates in {latent_path}")
        if feats.shape[1] != self.slat_feature_channels:
            raise ValueError(
                f"SLat features in {latent_path} have {feats.shape[1]} channels; "
                f"expected {self.slat_feature_channels}"
            )
        if not torch.isfinite(feats).all():
            raise ValueError(f"Non-finite SLat features in {latent_path}")
        if self.slat_normalization is not None:
            feats = (feats - self.slat_mean) / self.slat_std
        pack["slat_coords"] = coords
        pack["slat_feats"] = feats
        return pack

    @staticmethod
    def _pack_group(batch, group):
        sub_batch = [batch[index] for index in group]
        coords = []
        feats = []
        for batch_index, sample in enumerate(sub_batch):
            sample_coords = sample["slat_coords"]
            coords.append(
                torch.cat(
                    [
                        torch.full(
                            (sample_coords.shape[0], 1),
                            batch_index,
                            dtype=torch.int32,
                        ),
                        sample_coords,
                    ],
                    dim=-1,
                )
            )
            feats.append(sample["slat_feats"])

        pack = {
            "slat_x_0": SparseTensor(
                coords=torch.cat(coords, dim=0),
                feats=torch.cat(feats, dim=0),
            )
        }
        for key in sub_batch[0]:
            if key in {"slat_coords", "slat_feats"}:
                continue
            values = [sample[key] for sample in sub_batch]
            if isinstance(values[0], torch.Tensor):
                pack[key] = torch.stack(values)
            else:
                pack[key] = values
        return pack

    @classmethod
    def collate_fn(cls, batch, split_size=None):
        if split_size is None:
            groups = [list(range(len(batch)))]
        else:
            groups = load_balanced_group_indices(
                [sample["slat_coords"].shape[0] for sample in batch],
                split_size,
            )
            groups = [group for group in groups if group]
        packs = [cls._pack_group(batch, group) for group in groups]
        return packs[0] if split_size is None else packs
