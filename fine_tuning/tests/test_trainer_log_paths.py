import unittest

from trellis.trainers.base import get_named_log_paths


class TrainerLogPathTests(unittest.TestCase):
    def test_named_log_paths_use_output_directory_name_suffix(self):
        log_path, loss_path = get_named_log_paths("outputs/ss_enc_dec_fine_tune")

        self.assertEqual(log_path, "outputs/ss_enc_dec_fine_tune/log_ss_enc_dec_fine_tune.txt")
        self.assertEqual(loss_path, "outputs/ss_enc_dec_fine_tune/loss_ss_enc_dec_fine_tune.txt")

    def test_named_log_paths_ignore_trailing_separator(self):
        log_path, loss_path = get_named_log_paths("/tmp/run_a/")

        self.assertEqual(log_path, "/tmp/run_a/log_run_a.txt")
        self.assertEqual(loss_path, "/tmp/run_a/loss_run_a.txt")


if __name__ == "__main__":
    unittest.main()
