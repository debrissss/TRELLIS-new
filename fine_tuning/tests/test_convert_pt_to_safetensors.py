import unittest
from pathlib import Path

from fine_tuning import convert_pt_to_safetensors as module


class ConvertPtToSafetensorsTests(unittest.TestCase):
    def test_output_paths_preserve_dotted_prefix(self):
        prefix = Path("/tmp/denoiser_ema0.9999_step0050000")

        output_json, output_safetensors = module.get_output_paths(prefix)

        self.assertEqual(output_json, Path("/tmp/denoiser_ema0.9999_step0050000.json"))
        self.assertEqual(
            output_safetensors,
            Path("/tmp/denoiser_ema0.9999_step0050000.safetensors"),
        )

    def test_local_verification_uses_exact_written_files(self):
        prefix = Path("/tmp/denoiser_ema0.9999_step0050000")
        output_json, output_safetensors = module.get_output_paths(prefix)

        self.assertEqual(output_json.name, "denoiser_ema0.9999_step0050000.json")
        self.assertEqual(
            output_safetensors.name,
            "denoiser_ema0.9999_step0050000.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
