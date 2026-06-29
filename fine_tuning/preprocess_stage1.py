"""
人脸数据集预处理阶段一：Mesh 与相机规范化缩放及视角过滤。

本脚本对 FaceScape 3D 人脸 Mesh 进行尺度缩放和中心化平移，使其符合 [-0.5, 0.5]^3 的空间限制。
同时更新相机的内外参矩阵以保持重投影一致性，并过滤无效的相机视角。
"""

import os
import sys
import json
import argparse
import hashlib
import shutil
from pathlib import Path
import re
from typing import Optional, Tuple
import numpy as np
import trimesh

# 将项目根目录和当前目录加入 python 模块搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from fine_tuning.utils.logger import get_logger
from fine_tuning.utils.facescape_utils import get_facescape_subfolder, get_subject_paths, get_file_sha256
from camera_view_filter import compute_face_center

# 统一获取日志实例
logger = get_logger("preprocess_stage1")


def is_view_valid(
    camera_center: np.ndarray,
    face_center: np.ndarray,
    thresh_left_back: float,
    thresh_right_back: float,
    thresh_up: float,
    thresh_down: float,
) -> bool:
    """评估特定视角是否在允许的角度阈值范围内。

    使用 yaw (XZ平面夹角) 和 pitch (与水平面夹角) 对相机视角进行过滤。

    Args:
        camera_center (np.ndarray): 相机中心在世界坐标系下的 3D 坐标，形状为 (3,)。
        face_center (np.ndarray): 人脸中心 3D 坐标，形状为 (3,)。
        thresh_left_back (float): 左侧后方偏航角过滤阈值。
        thresh_right_back (float): 右侧后方偏航角过滤阈值。
        thresh_up (float): 上方俯仰角过滤阈值。
        thresh_down (float): 下方俯仰角过滤阈值。

    Returns:
        bool: 如果视角有效则返回 True，否则返回 False（丢弃）。
    """
    relative_pos = camera_center - face_center
    x, y, z = relative_pos[0], relative_pos[1], relative_pos[2]

    # 计算偏航角 (Yaw): 投影在 XZ 平面上，相对于 Z 轴正轴的夹角
    yaw_deg = np.degrees(np.arctan2(x, z))

    # 计算俯仰角 (Pitch): 相对于 XZ 水平平面的夹角
    horizontal_dist = np.sqrt(x**2 + z**2)
    pitch_deg = np.degrees(np.arctan2(y, horizontal_dist))

    # 判定 A：左侧后方 (Yaw <= -90°)
    if yaw_deg <= -90.0 and abs(yaw_deg) > thresh_left_back:
        return False
    # 判定 B：右侧后方 (Yaw >= 90°)
    if yaw_deg >= 90.0 and yaw_deg > thresh_right_back:
        return False
    # 判定 C：上方视角 (Pitch > 0°)
    if pitch_deg > 0.0 and pitch_deg > thresh_up:
        return False
    # 判定 D：下方视角 (Pitch < 0°)
    if pitch_deg < 0.0 and abs(pitch_deg) > thresh_down:
        return False

    return True


def filter_and_scan_expression(
    subject_id: str,
    expr_name: str,
    dataset_root: Path,
    center_mode: str,
    thresh_left_back: float,
    thresh_right_back: float,
    thresh_up: float,
    thresh_down: float,
) -> dict:
    """第一阶段（Pass 1）：对单个表情进行视角过滤与顶点跨度扫描。

    计算人脸中心，确定 T，计算 Mesh 外接跨度，并过滤有效视角（保留原始参数）。
    """
    closed_shapes_dir, aligned_params_dir = get_subject_paths(dataset_root, subject_id)
    mesh_path = closed_shapes_dir / f"{expr_name}.ply"
    camera_path = aligned_params_dir / expr_name / "params.json"

    if not camera_path.exists():
        alternative_params = list(aligned_params_dir.glob(f"**/{expr_name}/params.json"))
        if alternative_params:
            camera_path = alternative_params[0]
        else:
            raise FileNotFoundError("未能在路径下找到相机参数 params.json")

    # 1. 计算人脸中心，平移向量 T = -face_center
    with open(camera_path, "r", encoding="utf-8") as f:
        raw_params = json.load(f)

    camera_params = {}
    for k, v in raw_params.items():
        if "_" in k:
            parts = k.split("_", 1)
            if parts[0].isdigit():
                cam_id = int(parts[0])
                field_name = parts[1]
                if cam_id not in camera_params:
                    camera_params[cam_id] = {}
                camera_params[cam_id][field_name] = v

    face_center = compute_face_center(center_mode, mesh_path, camera_params)
    T = -face_center

    # 2. 载入网格，计算归中后的最大跨度
    mesh = trimesh.load(str(mesh_path), process=False)
    vertices_translated = mesh.vertices + T
    span = np.max(vertices_translated.max(axis=0) - vertices_translated.min(axis=0))
    del mesh
    del vertices_translated

    # 3. 视角过滤，筛选保留原始相机参数
    valid_raw_frames = []
    for cam_id, params in sorted(camera_params.items()):
        if "Rt" not in params:
            continue

        Rt = np.array(params["Rt"], dtype=np.float64)
        R = Rt[:, :3]
        t = Rt[:, 3]

        camera_center = -np.dot(R.T, t)

        # 视角过滤
        if is_view_valid(
            camera_center=camera_center,
            face_center=face_center,
            thresh_left_back=thresh_left_back,
            thresh_right_back=thresh_right_back,
            thresh_up=thresh_up,
            thresh_down=thresh_down,
        ):
            valid_raw_frames.append({
                "cam_id": cam_id,
                "params": params
            })

    return {
        "T": T,
        "span": span,
        "valid_raw_frames": valid_raw_frames,
        "mesh_path": mesh_path
    }


def scale_and_export_expression(
    subject_id: str,
    expr_name: str,
    output_dir: Path,
    S: float,
    T: np.ndarray,
    valid_raw_frames: list,
    mesh_path: Path,
) -> None:
    """第二阶段（Pass 2）：对单个表情网格进行规范化缩放并与有效帧一同导出。"""
    # 1. 计算原始 Mesh 的 SHA-256，决定最终的物理保存文件夹
    sha256 = get_file_sha256(mesh_path)
    render_dir = output_dir / "renders" / sha256
    render_dir.mkdir(parents=True, exist_ok=True)

    # 2. 读取并规范化 Mesh 尺度与坐标并写出
    logger.debug(f"正在读取并缩放人脸 Mesh: {mesh_path} ...")
    mesh = trimesh.load(str(mesh_path), process=False)
    mesh.vertices = S * (mesh.vertices + T)
    output_mesh_path = render_dir / "mesh.ply"
    mesh.export(str(output_mesh_path))
    del mesh
    logger.debug(f"人脸 Mesh 转换完成，已导出到 {output_mesh_path}")

    # 3. 遍历有效相机原始参数，应用缩放系数修正平移，转换至 Blender 并保存
    frames = []
    for frame_data in valid_raw_frames:
        cam_id = frame_data["cam_id"]
        params = frame_data["params"]

        Rt = np.array(params["Rt"], dtype=np.float64)
        R = Rt[:, :3]
        t = Rt[:, 3]

        # 修正相机平移向量: t' = S * (t - R * T)
        t_new = S * (t - np.dot(R, T))

        # 计算 FOV (camera_angle_x)
        width = params.get("width", 518)
        K = np.array(params.get("K", [[1000, 0, width/2], [0, 1000, 518/2], [0, 0, 1]]))
        f_x = K[0, 0]
        camera_angle_x = 2.0 * np.arctan(width / (2.0 * f_x))

        # 坐标系变换 (OpenCV -> Blender)
        R_cv = R.T
        t_cv = -np.dot(R_cv, t_new)
        transform_matrix = np.identity(4)
        transform_matrix[:3, 0] = R_cv[:, 0]
        transform_matrix[:3, 1] = -R_cv[:, 1]
        transform_matrix[:3, 2] = -R_cv[:, 2]
        transform_matrix[:3, 3] = t_cv

        frames.append({
            "file_path": f"normal_{cam_id:03d}.png",
            "camera_angle_x": float(camera_angle_x),
            "transform_matrix": transform_matrix.tolist()
        })

    # 4. 导出最终 transforms.json
    transforms_data = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": float(S),
        "offset": (S * T).tolist(),
        "frames": frames
    }

    output_json_path = render_dir / "transforms.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(transforms_data, f, indent=4)

    logger.debug(f"样本 #{subject_id} 表情 {expr_name} 完成处理，有效视角数: {len(frames)}，导出至 {output_json_path}")


def select_reference_translation(run_cache: dict, reference_id: str, reference_expr: Optional[str] = None) -> Tuple[np.ndarray, str, str]:
    """从扫描缓存中选择全局固定平移向量 T。

    T 的语义是预缩放平移量，最终网格坐标为 X' = S * (X + T)。
    """
    if reference_id not in run_cache:
        available_ids = sorted(run_cache.keys(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
        if not available_ids:
            raise ValueError("没有任何成功扫描的样本可用于选择 reference_id")
        fallback_id = available_ids[0]
        logger.warning(f"reference_id={reference_id} 未在成功扫描结果中找到，将回退使用 reference_id={fallback_id}")
        reference_id = fallback_id

    expr_cache = run_cache[reference_id]
    if reference_expr is None:
        available_exprs = sorted(expr_cache.keys())
        if not available_exprs:
            raise ValueError(f"reference_id={reference_id} 下没有任何成功扫描的表情")
        reference_expr = available_exprs[0]
    elif reference_expr not in expr_cache:
        available_exprs = sorted(expr_cache.keys())
        raise ValueError(
            f"reference_id={reference_id} 下未找到 reference_expr={reference_expr}；"
            f"可用表情: {available_exprs[:20]}"
        )

    return np.array(expr_cache[reference_expr]["T"], dtype=np.float64), reference_id, reference_expr


def filter_and_scan_worker(args_tuple) -> dict:
    """包装 filter_and_scan_expression，供多进程调用。"""
    subject_id, expr_name, dataset_root, center_mode, thresh_left_back, thresh_right_back, thresh_up, thresh_down = args_tuple
    try:
        res = filter_and_scan_expression(
            subject_id=subject_id,
            expr_name=expr_name,
            dataset_root=dataset_root,
            center_mode=center_mode,
            thresh_left_back=thresh_left_back,
            thresh_right_back=thresh_right_back,
            thresh_up=thresh_up,
            thresh_down=thresh_down
        )
        return {
            "success": True,
            "subject_id": subject_id,
            "expr_name": expr_name,
            "result": res
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "subject_id": subject_id,
            "expr_name": expr_name,
            "error": f"第一阶段失败: {str(e)}\n{traceback.format_exc()}"
        }
    finally:
        import gc
        gc.collect()


def scale_and_export_worker(args_tuple) -> dict:
    """包装 scale_and_export_expression，供多进程调用。"""
    subject_id, expr_name, output_dir, S, T, valid_raw_frames, mesh_path = args_tuple
    try:
        scale_and_export_expression(
            subject_id=subject_id,
            expr_name=expr_name,
            output_dir=output_dir,
            S=S,
            T=T,
            valid_raw_frames=valid_raw_frames,
            mesh_path=mesh_path
        )
        return {
            "success": True,
            "subject_id": subject_id,
            "expr_name": expr_name
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "subject_id": subject_id,
            "expr_name": expr_name,
            "error": f"第二阶段写出失败: {str(e)}\n{traceback.format_exc()}"
        }
    finally:
        import gc
        gc.collect()


def main() -> None:
    """主入口，解析人脸数据集路径、计算全局参数并进行数据处理。"""
    import time
    start_time = time.time()

    parser = argparse.ArgumentParser(description="阶段一：三维人脸与相机参数规范化缩放及视角过滤工具")

    parser.add_argument("--dataset_root", type=str, required=True,
                        help="FaceScape 原始数据集的根目录路径（应包含 001-020 文件夹）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出数据集的目标目录路径 (如 datasets/CustomFaceDataset)")

    # 归一化规范选项
    parser.add_argument("--scale", type=float, default=None,
                        help="外部指定固定缩放因子 S。若不指定，将依据 reference_id 样本动态计算")
    parser.add_argument("--translation", type=float, nargs=3, default=None,
                        help="外部指定固定平移向量 T (三个浮点数)。若不指定，将依据 reference_id 样本中心点计算")
    parser.add_argument("--reference_id", type=str, default="1",
                        help="基准参考样本 ID，用于计算全局固定的 S 和 T (默认: '1')")
    parser.add_argument("--reference_expr", type=str, default=None,
                        help="基准参考表情名，用于计算全局固定 T。若不指定，将使用 reference_id 下按名称排序的第一个表情")
    parser.add_argument("--target_extent", type=float, default=0.95,
                        help="自动计算缩放系数时，全局最大单轴跨度的目标大小（默认: 0.95）")
    parser.add_argument("--center_mode", type=str, default="camera_intersection",
                        choices=["origin", "mesh_mean", "mesh_bbox", "camera_intersection"],
                        help="人脸中心点计算模式（默认: camera_intersection）")

    # 视角过滤阈值选项 (度)
    parser.add_argument("--thresh_left_back", type=float, default=100.0,
                        help="左侧后方偏航角 Yaw 过滤阈值（默认: 100.0）")
    parser.add_argument("--thresh_right_back", type=float, default=100.0,
                        help="右侧后方偏航角 Yaw 过滤阈值（默认: 100.0）")
    parser.add_argument("--thresh_up", type=float, default=40.0,
                        help="上方俯仰角 Pitch 过滤阈值（默认: 40.0）")
    parser.add_argument("--thresh_down", type=float, default=40.0,
                        help="下方俯仰角 Pitch 过滤阈值（默认: 40.0）")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="多进程工作核心数（默认: max(1, os.cpu_count() - 2)）")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    if not dataset_root.exists():
        logger.error(f"数据集根目录不存在: {dataset_root}")
        sys.exit(1)

    # 计算自适应的进程数
    if args.num_workers is not None:
        num_workers = max(1, args.num_workers)
    else:
        num_workers = max(1, os.cpu_count() - 2)
    logger.info(f"启用多进程并行加速，工作进程数: {num_workers}")

    # 2. 遍历并扫描全部样本目录
    subject_dirs = []

    # 尝试扁平结构 (例如直接在 dataset_root 下)
    flat_shapes_dir = dataset_root / "closed_shapes_meshlib"
    if flat_shapes_dir.exists():
        subject_dirs.extend([d for d in flat_shapes_dir.iterdir() if d.is_dir()])

    # 尝试标准 FaceScape 分段结构 (匹配类似 001-020 的三位数字范围子目录)
    for item in dataset_root.iterdir():
        if item.is_dir() and re.match(r"^\d{3}-\d{3}$", item.name):
            sub_shapes_dir = item / "closed_shapes_meshlib"
            if sub_shapes_dir.exists():
                subject_dirs.extend([d for d in sub_shapes_dir.iterdir() if d.is_dir()])

    if not subject_dirs:
        logger.error(f"在数据集根目录 {dataset_root} 下未检测到任何 closed_shapes_meshlib 样本！")
        sys.exit(1)

    # 依据数字大小排序（处理 '1', '2', ..., '20' 等非零填充名字）
    subject_dirs = sorted(subject_dirs, key=lambda x: int(x.name) if x.name.isdigit() else 999)

    # 1. 视角过滤与跨度扫描（第一阶段，Pass 1）
    run_cache = {}
    max_L = 0.0
    span_records = []
    failures = []

    # 收集并扁平化所有待处理任务
    tasks = []
    for s_dir in subject_dirs:
        subject_id = s_dir.name
        closed_shapes_dir, _ = get_subject_paths(dataset_root, subject_id)
        if not closed_shapes_dir.exists():
            continue
        ply_files = list(closed_shapes_dir.glob("*.ply"))
        for mesh_path in ply_files:
            tasks.append((subject_id, mesh_path.stem, mesh_path))

    logger.info(f"开始第一阶段：全局视角过滤与几何跨度扫描（共 {len(tasks)} 个样本）...")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    # 构造 Pass 1 任务包
    pass1_args = [
        (
            subject_id,
            expr_name,
            dataset_root,
            args.center_mode,
            args.thresh_left_back,
            args.thresh_right_back,
            args.thresh_up,
            args.thresh_down
        )
        for subject_id, expr_name, _ in tasks
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(filter_and_scan_worker, item): (item[0], item[1])
            for item in pass1_args
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="视角过滤与跨度扫描中"):
            subject_id, expr_name = futures[future]
            try:
                res = future.result()
                if res["success"]:
                    if subject_id not in run_cache:
                        run_cache[subject_id] = {}
                    run_cache[subject_id][expr_name] = res["result"]
                    span = float(res["result"]["span"])
                    span_records.append({
                        "subject_id": subject_id,
                        "expr_name": expr_name,
                        "span": span
                    })
                    max_L = max(max_L, span)
                else:
                    failures.append({
                        "subject_id": subject_id,
                        "expr_name": expr_name,
                        "error": res["error"]
                    })
            except Exception as e:
                import traceback
                failures.append({
                    "subject_id": subject_id,
                    "expr_name": expr_name,
                    "error": f"进程执行异常: {str(e)}\n{traceback.format_exc()}"
                })

    # 2. 计算全局唯一的缩放因子 S 与固定平移向量 T
    if args.scale is not None and args.translation is not None:
        S = args.scale
        T_fixed = np.array(args.translation, dtype=np.float64)
        if T_fixed.shape != (3,):
            logger.error("--translation 必须包含三个浮点数")
            sys.exit(1)
        for subject_id in run_cache:
            for expr_name in run_cache[subject_id]:
                run_cache[subject_id][expr_name]["T"] = T_fixed
        logger.info(
            f"直接使用外部指定的固定归一化参数: S = {S}, T = {T_fixed}。"
            "注意 T 是预缩放平移量，导出公式为 X' = S * (X + T)。"
        )
    elif args.scale is not None or args.translation is not None:
        logger.error("--scale 与 --translation 必须同时指定，或同时不指定以启用自动全局归一化")
        sys.exit(1)
    else:
        if max_L == 0.0:
            logger.error("没有任何有效的三维网格跨度可以计算缩放系数！")
            sys.exit(1)
        if args.target_extent <= 0:
            logger.error("--target_extent 必须大于 0")
            sys.exit(1)
        span_values = np.array([item["span"] for item in span_records], dtype=np.float64)
        max_span_record = max(span_records, key=lambda item: item["span"])
        S = args.target_extent / max_L
        try:
            T_fixed, resolved_reference_id, resolved_reference_expr = select_reference_translation(
                run_cache=run_cache,
                reference_id=args.reference_id,
                reference_expr=args.reference_expr,
            )
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        for subject_id in run_cache:
            for expr_name in run_cache[subject_id]:
                run_cache[subject_id][expr_name]["T"] = T_fixed
        logger.info(
            f"第一阶段完成。最大几何单轴跨度 L_max = {max_L:.6f}，"
            f"目标跨度 = {args.target_extent:.6f}，全局安全缩放因子 S = {S:.6f}。"
        )
        logger.info(
            f"L_max 来源: subject_id={max_span_record['subject_id']}, "
            f"expr_name={max_span_record['expr_name']}。"
        )
        logger.info(
            f"跨度统计: min={np.min(span_values):.6f}, p50={np.percentile(span_values, 50):.6f}, "
            f"p95={np.percentile(span_values, 95):.6f}, max={np.max(span_values):.6f}。"
        )
        logger.info(
            f"使用全局固定平移 T = {T_fixed}，来源 reference_id={resolved_reference_id}, "
            f"reference_expr={resolved_reference_expr}。"
        )

    # 3. 规范化缩放与结果写出（第二阶段，Pass 2）
    logger.info("开始第二阶段：全局人脸网格缩放与标准相机参数导出...")

    processed_subjects = 0
    # 仅遍历在第一阶段缓存成功的任务
    valid_tasks = []
    for subject_id in run_cache:
        for expr_name in run_cache[subject_id]:
            cache = run_cache[subject_id][expr_name]
            if len(cache["valid_raw_frames"]) > 0:
                valid_tasks.append((subject_id, expr_name, cache))
            else:
                failures.append({
                    "subject_id": subject_id,
                    "expr_name": expr_name,
                    "error": "该样本没有任何有效相机视角通过偏航角/俯仰角过滤被保留"
                })

    # 构造 Pass 2 任务包
    pass2_args = [
        (
            subject_id,
            expr_name,
            output_dir,
            S,
            cache["T"],
            cache["valid_raw_frames"],
            cache["mesh_path"]
        )
        for subject_id, expr_name, cache in valid_tasks
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(scale_and_export_worker, item): (item[0], item[1])
            for item in pass2_args
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="规范化缩放与导出中"):
            subject_id, expr_name = futures[future]
            try:
                res = future.result()
                if res["success"]:
                    processed_subjects += 1
                else:
                    failures.append({
                        "subject_id": subject_id,
                        "expr_name": expr_name,
                        "error": res["error"]
                    })
            except Exception as e:
                import traceback
                failures.append({
                    "subject_id": subject_id,
                    "expr_name": expr_name,
                    "error": f"进程执行异常: {str(e)}\n{traceback.format_exc()}"
                })

    # 4. 统计结果和耗时日志
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{minutes} 分 {seconds} 秒" if minutes > 0 else f"{seconds} 秒"

    logger.info(f"阶段一预处理圆满结束！成功处理并保存 {processed_subjects} 个表情样本，总耗时 {time_str}。")
    logger.info(f"应用的全局缩放因子 S = {S:.6f}。")

    if failures:
        logger.error(f"====== 共检测到 {len(failures)} 个表情样本处理失败 ======")
        for fail in failures:
            logger.error(f"样本 #{fail['subject_id']} 表情 {fail['expr_name']} 失败原因:")
            logger.error(fail["error"])
            logger.error("=" * 60)
        logger.error("==================================================")
    else:
        logger.info("所有表情样本全部处理成功，无失败记录。")


if __name__ == "__main__":
    main()
