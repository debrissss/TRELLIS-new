import tempfile
import unittest
from pathlib import Path

import pandas as pd

from eval.prepare_ss_eval_dataset import create_eval_dataset


class PrepareSparseStructureEvalDatasetTests(unittest.TestCase):
    def test_creates_metadata_subset_and_voxel_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "eval"
            voxels = source / "voxels"
            voxels.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"sha256": "a", "aesthetic_score": 5.0, "voxelized": True, "split": "test"},
                    {"sha256": "b", "aesthetic_score": 4.0, "voxelized": True, "split": "test"},
                    {"sha256": "c", "aesthetic_score": 5.0, "voxelized": False, "split": "test"},
                    {"sha256": "d", "aesthetic_score": 5.0, "voxelized": True, "split": "test"},
                ]
            ).to_csv(source / "metadata.csv", index=False)
            (voxels / "a.ply").write_text("ply\n", encoding="utf-8")
            (voxels / "d.ply").write_text("ply\n", encoding="utf-8")

            selected = create_eval_dataset(
                source_root=source,
                output_root=output,
                num_samples=2,
                seed=7,
                min_aesthetic_score=4.5,
                replace=False,
            )

            metadata = pd.read_csv(output / "metadata.csv")
            self.assertEqual(set(metadata["sha256"]), {"a", "d"})
            self.assertEqual(selected, ["a", "d"])
            self.assertTrue((output / "voxels").is_symlink())
            self.assertEqual((output / "voxels").resolve(), voxels.resolve())

    def test_refuses_to_sample_more_than_available_without_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            voxels = source / "voxels"
            voxels.mkdir(parents=True)
            pd.DataFrame(
                [{"sha256": "a", "aesthetic_score": 5.0, "voxelized": True}]
            ).to_csv(source / "metadata.csv", index=False)
            (voxels / "a.ply").write_text("ply\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                create_eval_dataset(
                    source_root=source,
                    output_root=root / "eval",
                    num_samples=2,
                    seed=7,
                    min_aesthetic_score=None,
                    replace=False,
                )


if __name__ == "__main__":
    unittest.main()
