import csv
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from eval.compare_slat_metrics import build_comparison_rows
from eval.prepare_facescape_eval_subset import prepare_eval_subset


class PrepareFacescapeEvalSubsetTest(unittest.TestCase):
    def test_prepare_eval_subset_filters_valid_samples_and_writes_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            output_dir = root / "eval"
            feature_model = "dinov2_vitl14_reg"
            rows = [
                {
                    "sha256": "sha-a",
                    f"feature_{feature_model}": "True",
                    "aesthetic_score": "5.0",
                    "num_voxels": "120",
                },
                {
                    "sha256": "sha-b",
                    f"feature_{feature_model}": "True",
                    "aesthetic_score": "5.0",
                    "num_voxels": "121",
                },
                {
                    "sha256": "sha-missing-feature",
                    f"feature_{feature_model}": "True",
                    "aesthetic_score": "5.0",
                    "num_voxels": "122",
                },
                {
                    "sha256": "sha-flag-false",
                    f"feature_{feature_model}": "False",
                    "aesthetic_score": "5.0",
                    "num_voxels": "123",
                },
            ]
            source_dir.mkdir()
            with (source_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            for sha in ["sha-a", "sha-b", "sha-missing-feature", "sha-flag-false"]:
                render_dir = source_dir / "renders" / sha
                render_dir.mkdir(parents=True)
                (render_dir / "transforms.json").write_text('{"frames": []}', encoding="utf-8")
            feature_dir = source_dir / "features" / feature_model
            feature_dir.mkdir(parents=True)
            for sha in ["sha-a", "sha-b", "sha-flag-false"]:
                np.savez(feature_dir / f"{sha}.npz", indices=np.zeros((1, 3)), patchtokens=np.zeros((1, 2)))

            result = prepare_eval_subset(
                source_dir=source_dir,
                output_dir=output_dir,
                num_samples=2,
                seed=7,
                feature_model=feature_model,
                copy_files=False,
                overwrite=False,
            )

            self.assertEqual(result.selected_count, 2)
            selected = (output_dir / "selected_sha256.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(sorted(selected), ["sha-a", "sha-b"])
            self.assertTrue((output_dir / "renders" / selected[0]).is_symlink())
            self.assertTrue((output_dir / "features" / feature_model / f"{selected[0]}.npz").is_symlink())

            with (output_dir / "metadata.csv").open(encoding="utf-8") as f:
                metadata_rows = list(csv.DictReader(f))
            self.assertEqual(sorted(row["sha256"] for row in metadata_rows), ["sha-a", "sha-b"])

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_dir"], str(source_dir))
            self.assertEqual(manifest["feature_model"], feature_model)
            self.assertEqual(manifest["storage"], "symlink")


class CompareSLatMetricsTest(unittest.TestCase):
    def test_build_comparison_rows_flattens_summary_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            run_a.mkdir()
            run_b.mkdir()
            (run_a / "summary.json").write_text(
                json.dumps({
                    "num_records": 2,
                    "failed_count": 0,
                    "metrics": {
                        "l1": {"mean": 0.10, "p50": 0.09, "p90": 0.12},
                        "psnr": {"mean": 24.0, "p50": 24.1, "p90": 25.0},
                    },
                }),
                encoding="utf-8",
            )
            (run_b / "summary.json").write_text(
                json.dumps({
                    "num_records": 3,
                    "failed_count": 1,
                    "metrics": {
                        "l1": {"mean": 0.08, "p50": 0.07, "p90": 0.11},
                        "lpips": {"mean": 0.04},
                    },
                }),
                encoding="utf-8",
            )

            rows = build_comparison_rows({"a": run_a, "b": run_b})

            self.assertEqual(rows[0]["name"], "a")
            self.assertEqual(rows[0]["num_records"], 2)
            self.assertEqual(rows[0]["mean_l1"], 0.10)
            self.assertEqual(rows[0]["p90_l1"], 0.12)
            self.assertEqual(rows[1]["name"], "b")
            self.assertEqual(rows[1]["failed_count"], 1)
            self.assertEqual(rows[1]["mean_lpips"], 0.04)


if __name__ == "__main__":
    unittest.main()
