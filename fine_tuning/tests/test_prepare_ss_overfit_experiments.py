import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fine_tuning import prepare_ss_overfit_experiments as module


class PrepareSsOverfitExperimentsTest(unittest.TestCase):
    def test_select_neutral_rows_requires_assets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"sha256": "sha-a", "captions": "1_1_neutral", "cond_rendered": "True", "ss_latent_ss_enc_conv3d_16l8_fp16": "True"},
                {"sha256": "sha-b", "captions": "2_1_neutral", "cond_rendered": "True", "ss_latent_ss_enc_conv3d_16l8_fp16": "True"},
                {"sha256": "sha-c", "captions": "1_2_smile", "cond_rendered": "True", "ss_latent_ss_enc_conv3d_16l8_fp16": "True"},
            ]
            for sha in ["sha-a", "sha-c"]:
                (root / "renders_cond" / sha).mkdir(parents=True)
                (root / "renders_cond" / sha / "17.png").write_text("normal", encoding="utf-8")
                (root / "ss_latents" / module.LATENT_MODEL).mkdir(parents=True, exist_ok=True)
                (root / "ss_latents" / module.LATENT_MODEL / f"{sha}.npz").write_text("latent", encoding="utf-8")

            selected = module.select_neutral_rows(rows, root, count=4)

            self.assertEqual([row["sha256"] for row in selected], ["sha-a"])

    def test_prepare_subset_writes_metadata_and_symlinks_assets(self):
        with TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            subset_root = Path(tmp) / "subset"
            sha = "sha-a"
            render_source = source_root / "renders_cond" / sha
            latent_source = source_root / "ss_latents" / module.LATENT_MODEL / f"{sha}.npz"
            render_source.mkdir(parents=True)
            render_source.joinpath("17.png").write_text("normal", encoding="utf-8")
            latent_source.parent.mkdir(parents=True)
            latent_source.write_text("latent", encoding="utf-8")
            row = {"sha256": sha, "captions": "1_1_neutral", "cond_rendered": "True", "ss_latent_ss_enc_conv3d_16l8_fp16": "True"}

            module.prepare_subset(source_root, subset_root, [row])

            with (subset_root / "metadata.csv").open(encoding="utf-8") as f:
                metadata_rows = list(csv.DictReader(f))
            self.assertEqual(metadata_rows[0]["sha256"], sha)
            self.assertTrue((subset_root / "renders_cond" / sha).is_symlink())
            self.assertTrue((subset_root / "ss_latents" / module.LATENT_MODEL / f"{sha}.npz").is_symlink())

    def test_build_overfit_config_sets_small_run_parameters(self):
        base_config = {
            "trainer": {
                "args": {
                    "max_steps": 40000,
                    "batch_size_per_gpu": 16,
                    "batch_split": 8,
                    "i_save": 500,
                    "i_sample": 2000,
                }
            }
        }

        config = module.build_overfit_config(base_config, max_steps=3000, sample_count=1)

        trainer_args = config["trainer"]["args"]
        self.assertEqual(trainer_args["max_steps"], 3000)
        self.assertEqual(trainer_args["batch_size_per_gpu"], 1)
        self.assertEqual(trainer_args["batch_split"], 1)
        self.assertEqual(trainer_args["dataloader_num_workers"], 0)
        self.assertEqual(trainer_args["dataloader_drop_last"], False)
        self.assertEqual(trainer_args["dataloader_persistent_workers"], False)
        self.assertEqual(trainer_args["prefetch_data"], False)
        self.assertEqual(trainer_args["i_save"], 500)
        self.assertEqual(trainer_args["i_sample"], 500)
        self.assertEqual(trainer_args["i_log"], 10)
        self.assertEqual(trainer_args["i_print"], 10)

    def test_write_run_script_contains_config_data_and_output_paths(self):
        with TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "run.sh"
            module.write_run_script(
                script_path=script_path,
                config_path=Path("/cfg.json"),
                data_dir=Path("/data"),
                output_dir=Path("/out"),
            )

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--config /cfg.json", content)
            self.assertIn("--data_dir /data", content)
            self.assertIn("--output_dir /out", content)
            self.assertIn("CONDA_ENV=", content)
            self.assertIn('PYTHON="${PYTHON:-/root/autodl-tmp/mamba_envs/trellis5090/bin/python}"', content)
            self.assertIn("export OMP_NUM_THREADS=8", content)


if __name__ == "__main__":
    unittest.main()
