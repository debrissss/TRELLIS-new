import os
import unittest

from trellis.trainers.base import make_output_log_paths


class TrainerLogPathsTest(unittest.TestCase):
    def test_make_output_log_paths_uses_output_directory_basename(self):
        log_path, loss_path = make_output_log_paths("output/slat_enc_dec")

        self.assertEqual(log_path, os.path.join("output/slat_enc_dec", "log_slat_enc_dec.txt"))
        self.assertEqual(loss_path, os.path.join("output/slat_enc_dec", "loss_slat_enc_dec.txt"))

    def test_make_output_log_paths_ignores_trailing_separator(self):
        log_path, loss_path = make_output_log_paths("output/slat_enc_dec/")

        self.assertEqual(log_path, os.path.join("output/slat_enc_dec/", "log_slat_enc_dec.txt"))
        self.assertEqual(loss_path, os.path.join("output/slat_enc_dec/", "loss_slat_enc_dec.txt"))


if __name__ == "__main__":
    unittest.main()
