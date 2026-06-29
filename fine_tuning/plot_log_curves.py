import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


NumericSeries = Dict[str, List[Tuple[int, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse TRELLIS log.txt and plot every numeric metric as a subplot."
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to log.txt. Each non-empty line must be formatted as 'step: {json}'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for per-metric PNG files. Defaults to log_curves next to the log file.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI. Defaults to 150.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Single figure width in inches. Defaults to 16.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Single figure height in inches. Defaults to 4.5.",
    )
    return parser.parse_args()


def is_numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_numeric_metrics(data: dict, prefix: str = "") -> Iterable[Tuple[str, float]]:
    for key, value in data.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from flatten_numeric_metrics(value, path)
        elif is_numeric(value):
            yield path, float(value)


def parse_log_line(line: str, line_number: int) -> Tuple[int, dict]:
    if ":" not in line:
        raise ValueError(f"Line {line_number}: expected 'step: {{json}}' format.")

    step_text, payload_text = line.split(":", 1)
    try:
        step = int(step_text.strip())
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: invalid step value {step_text!r}.") from exc

    try:
        payload = json.loads(payload_text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Line {line_number}: invalid JSON payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Line {line_number}: JSON payload must be an object.")

    return step, payload


def parse_log(log_path: Path) -> NumericSeries:
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")
    if not log_path.is_file():
        raise ValueError(f"log path is not a file: {log_path}")

    series: NumericSeries = {}
    with log_path.open("r", encoding="utf-8") as log_file:
        for line_number, raw_line in enumerate(log_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            step, payload = parse_log_line(line, line_number)
            for metric, value in flatten_numeric_metrics(payload):
                series.setdefault(metric, []).append((step, value))

    return series


def require_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "trellis_matplotlib_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot log curves. Install it and rerun this script."
        ) from exc

    return plt


def require_ticker():
    try:
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot log curves. Install it and rerun this script."
        ) from exc

    return AutoMinorLocator, MaxNLocator


def metric_to_filename(metric: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric).strip("_")
    return f"{filename or 'metric'}.png"


def plot_series(
    series: NumericSeries,
    output_dir: Path,
    dpi: int,
    fig_width: float | None,
    fig_height: float | None,
) -> List[Path]:
    metrics = sorted(metric for metric, points in series.items() if points)
    if not metrics:
        raise ValueError("No numeric metrics found in the log file.")

    plt = require_matplotlib()
    AutoMinorLocator, MaxNLocator = require_ticker()
    width = fig_width if fig_width is not None else 16.0
    height = fig_height if fig_height is not None else 4.5
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for metric in metrics:
        points = sorted(series[metric], key=lambda item: item[0])
        steps = [step for step, _ in points]
        values = [value for _, value in points]

        fig, axis = plt.subplots(figsize=(width, height))
        axis.plot(steps, values, marker="o", markersize=2, linewidth=1)
        axis.set_title(metric, fontsize=11)
        axis.set_xlabel("step")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=25))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=25, integer=True))
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis.yaxis.set_minor_locator(AutoMinorLocator(5))
        axis.grid(True, which="major", linewidth=0.5, alpha=0.55)
        axis.grid(True, which="minor", linewidth=0.3, alpha=0.25)

        fig.tight_layout()
        output_path = output_dir / metric_to_filename(metric)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    opt = parse_args()
    output_dir = opt.output if opt.output is not None else opt.log.with_name("log_curves")
    try:
        series = parse_log(opt.log)
        output_paths = plot_series(
            series=series,
            output_dir=output_dir,
            dpi=opt.dpi,
            fig_width=opt.fig_width,
            fig_height=opt.fig_height,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from None

    print(f"[INFO] Wrote {len(output_paths)} metric curve PNG files to {output_dir}")


if __name__ == "__main__":
    main()
