"""Preferred entry point for SS ControlNet timestep-schedule evaluation."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.eval_face_scan_ControlNet_ss_scale_sweep import main


if __name__ == "__main__":
    main()
