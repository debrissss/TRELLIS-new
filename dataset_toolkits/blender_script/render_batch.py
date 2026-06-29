import argparse
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace


def _load_render_module():
    render_path = os.path.join(os.path.dirname(__file__), "render.py")
    spec = importlib.util.spec_from_file_location("trellis_blender_render", render_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _namespace_from_job(job, args):
    return SimpleNamespace(
        views=json.dumps(job["views"]),
        object=job["object"],
        output_folder=job["output_folder"],
        resolution=args.resolution,
        engine=args.engine,
        geo_mode=args.geo_mode,
        save_depth=args.save_depth,
        save_normal=args.save_normal,
        save_albedo=args.save_albedo,
        save_mist=args.save_mist,
        split_normal=args.split_normal,
        save_mesh=args.save_mesh,
        skip_normalize=args.skip_normalize,
        profile=args.profile,
        profile_disable_denoise=args.profile_disable_denoise,
        profile_no_write=args.profile_no_write,
    )


def main(args):
    render_module = _load_render_module()
    with open(args.jobs, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    records = []
    for index, job in enumerate(jobs):
        start = time.perf_counter()
        sha256 = job.get("sha256", "")
        try:
            print(f"[BATCH] start index={index} sha256={sha256}", flush=True)
            render_module.main(_namespace_from_job(job, args))
            records.append({
                "sha256": sha256,
                "rendered": True,
                "wall_seconds": time.perf_counter() - start,
            })
            print(f"[BATCH] done index={index} sha256={sha256}", flush=True)
        except Exception as e:
            records.append({
                "sha256": sha256,
                "rendered": False,
                "wall_seconds": time.perf_counter() - start,
                "error": str(e),
            })
            print(f"[BATCH] failed index={index} sha256={sha256} error={e}", flush=True)

    with open(args.records, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render multiple objects in one Blender process.")
    parser.add_argument("--jobs", type=str, required=True)
    parser.add_argument("--records", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--geo_mode", action="store_true")
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--save_normal", action="store_true")
    parser.add_argument("--save_albedo", action="store_true")
    parser.add_argument("--save_mist", action="store_true")
    parser.add_argument("--split_normal", action="store_true")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--skip_normalize", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_disable_denoise", action="store_true")
    parser.add_argument("--profile_no_write", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:]
    main(parser.parse_args(argv))
