"""
人脸数据集预处理阶段二：法线图背景提取、条件 RGBA 图像生成及姿态对齐。

本脚本读取 facescape_filter_views.py 生成的 view_filters/{sha256}/transforms.json，
定位其对应的原始 3 通道法线图。
根据物理特征过滤纯白背景区域生成 Alpha 通道，并将转换后的 4 通道 RGBA 图像与相机参数对齐输出。
"""

import os
import sys
import json
import argparse
import re
import traceback
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# 将项目根目录和当前目录加入 python 模块搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from fine_tuning.utils.logger import get_logger
from fine_tuning.utils.facescape_utils import get_facescape_subfolder, get_subject_paths, get_file_sha256

# 统一获取日志实例
logger = get_logger("preprocess_stage2")


def process_subject_normals(
    subject_id: str,
    expr_name: str,
    mesh_path: Path,
    dataset_root: Path,
    output_dir: Path,
    target_size: int = 1024,
    num_threads: int = 8,
    fmt: str = "png",
    webp_quality: int = 101,
) -> int:
    """处理单个人脸样本特定表情的全部法线图，将白色背景融合为透明 Alpha 遮罩。

    Args:
        subject_id (str): 样本 ID。
        expr_name (str): 表情名称。
        mesh_path (Path): 原始 Mesh 路径，用于计算 SHA-256 哈希定位输出目录。
        dataset_root (Path): 原始数据集根目录。
        output_dir (Path): 规范化数据集的输出目录。
        target_size (int): 等比缩放目标长边分辨率。如果为 0 或 None，则保持原始分辨率。

    Returns:
        int: 成功处理的法线视角图像数。
    """
    # 限制 OpenCV 自身的内部多线程，防止嵌套多线程冲突导致 CPU 超配
    cv2.setNumThreads(1)

    # 1. 计算哈希定位视角过滤结果子目录
    sha256 = get_file_sha256(mesh_path)
    view_filter_dir = output_dir / "view_filters" / sha256
    transforms_path = view_filter_dir / "transforms.json"

    if not transforms_path.exists():
        logger.debug(f"样本 #{subject_id} 表情 {expr_name} 未生成 view_filters transforms.json，跳过法线转换。")
        return 0

    # 创建独立的阶段二条件图导出目录
    render_cond_dir = output_dir / "renders_cond" / sha256
    render_cond_dir.mkdir(parents=True, exist_ok=True)

    # 2. 读取 view_filters transforms.json 以获取有效视角列表
    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms_data = json.load(f)

    frames = transforms_data.get("frames", [])
    if not frames:
        logger.debug(f"样本 #{subject_id} 表情 {expr_name} 没有有效帧视角，跳过。")
        return 0

    # 3. 定位原始法线图所在的父目录
    subfolder = get_facescape_subfolder(subject_id)
    if subfolder:
        raw_normal_dir = dataset_root / subfolder / "normals" / subject_id / expr_name
    else:
        raw_normal_dir = dataset_root / "normals" / subject_id / expr_name

    if not raw_normal_dir.exists():
        raise FileNotFoundError(f"未能在路径下找到原始法线图目录: {raw_normal_dir}")

    # 4. 使用线程池并发处理法线图
    from concurrent.futures import ThreadPoolExecutor

    def process_single_frame(frame) -> bool:
        # 提取目标文件名，如 normal_001.png
        target_name = frame["file_path"]
        
        # 从文件名解析出整数 cam_id（例如 normal_005.png -> 5）
        match = re.search(r"normal_(\d+)\.png", target_name)
        if not match:
            logger.warning(f"无法从有效帧路径解析相机 ID: {target_name}，跳过该帧")
            return False
            
        cam_id = int(match.group(1))
        
        # 拼接原始无前导零的法线图路径，例如 1.png, 5.png
        raw_normal_path = raw_normal_dir / f"{cam_id}.png"
        if not raw_normal_path.exists():
            raise FileNotFoundError(f"未找到原始视角相机 #{cam_id:03d} 的法线图片: {raw_normal_path}")

        # 读取并进行 RGBA 背景融合
        logger.debug(f"正在转换法线图: {raw_normal_path} -> RGBA ({fmt})")
        
        img_bgr = cv2.imread(str(raw_normal_path))
        if img_bgr is None:
            raise FileNotFoundError(f"无法使用 OpenCV 读取法线图片: {raw_normal_path}")

        # 保持长宽比等比缩放，限制长边不超过 target_size
        if target_size is not None and target_size > 0:
            h, w = img_bgr.shape[:2]
            max_dim = max(w, h)
            if max_dim > target_size:
                scale = target_size / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 物理保底判定：B、G、R 通道均大于 254 则判为纯白背景 (Alpha = 0)
        min_val = np.minimum(np.minimum(img_bgr[:, :, 0], img_bgr[:, :, 1]), img_bgr[:, :, 2])
        is_bg = min_val > 254
        
        # 在 OpenCV 层直接融合成 RGBA (BGRA)
        img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        img_bgra[is_bg, 3] = 0
        
        # 根据格式参数保存
        if fmt.lower() == "webp":
            output_target_name = target_name.replace(".png", ".webp")
            output_normal_path = render_cond_dir / output_target_name
            cv2.imwrite(str(output_normal_path), img_bgra, [cv2.IMWRITE_WEBP_QUALITY, webp_quality])
        else:
            output_normal_path = render_cond_dir / target_name
            cv2.imwrite(str(output_normal_path), img_bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        # 显式释放资源
        del img_bgr
        del img_bgra
        del min_val
        del is_bg

        logger.debug(f"法线图转换完成已保存: {output_normal_path}")
        return True

    # 限制并发线程数，使用自适应的线程数，最大不超过当前样本视角总数
    max_threads = min(num_threads, len(frames))
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        results = list(pool.map(process_single_frame, frames))
    
    processed_views = sum(1 for r in results if r)

    # 保存条件相机姿态索引文件到 renders_cond 子目录下以供训练 Dataset 读取
    if processed_views > 0:
        # 如果是 webp 格式，将 transforms_data 中的文件名后缀也统一修改为 .webp 以匹配存储
        if fmt.lower() == "webp":
            for frame in frames:
                if "file_path" in frame:
                    frame["file_path"] = frame["file_path"].replace(".png", ".webp")
                
        output_transforms_path = render_cond_dir / "transforms.json"
        with open(output_transforms_path, "w", encoding="utf-8") as f:
            json.dump(transforms_data, f, indent=4)
        logger.debug(f"条件 transforms.json 导出完成: {output_transforms_path}")

    return processed_views


def process_subject_normals_worker(args_tuple) -> dict:
    """包装 process_subject_normals，供多进程调用。"""
    subject_id, expr_name, mesh_path, dataset_root, output_dir, target_size, num_threads, fmt, webp_quality = args_tuple
    try:
        views_cnt = process_subject_normals(
            subject_id=subject_id,
            expr_name=expr_name,
            mesh_path=mesh_path,
            dataset_root=dataset_root,
            output_dir=output_dir,
            target_size=target_size,
            num_threads=num_threads,
            fmt=fmt,
            webp_quality=webp_quality
        )
        return {
            "success": True,
            "subject_id": subject_id,
            "expr_name": expr_name,
            "views_cnt": views_cnt
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "subject_id": subject_id,
            "expr_name": expr_name,
            "error": f"阶段二失败: {str(e)}\n{traceback.format_exc()}"
        }
    finally:
        import gc
        gc.collect()


def main() -> None:
    """主入口，遍历人脸数据集、读取阶段一结果并执行法线图背景透明化转换。"""
    import time
    start_time = time.time()

    parser = argparse.ArgumentParser(description="阶段二：法线图背景提取、条件 RGBA 生成与姿态对齐工具")
    
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="FaceScape 原始数据集的根目录路径（应包含 001-020 文件夹）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出数据集的目标目录路径 (如 datasets/CustomFaceDataset)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="多进程工作核心数（默认: max(1, os.cpu_count() - 2)）")
    parser.add_argument("--target_size", type=int, default=1024,
                        help="等比缩放目标长边分辨率。如果为 0，则保持原始分辨率（默认: 1024）")
    parser.add_argument("--format", type=str, default="png", choices=["png", "webp"],
                        help="输出法线图像的格式 (默认: png)")
    parser.add_argument("--webp_quality", type=int, default=101,
                        help="WebP 编码质量 (1-100 为有损，>100 为无损，默认: 101)")
    parser.add_argument("--instances", type=str, default=None,
                        help="只处理指定 sha256，支持逗号分隔列表或每行一个 sha256 的文件")
    
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
    
    # 计算每个工作进程内线程池的自适应线程数，防止 CPU 线程超配
    total_cpus = os.cpu_count()
    threads_per_worker = max(1, total_cpus // num_workers)
    threads_per_worker = min(threads_per_worker, 8)
    logger.info(f"每个工作进程内的最大并发线程数（自适应限制）: {threads_per_worker}")
        
    if not output_dir.exists():
        logger.error(f"规范化输出目录不存在（请先运行阶段一处理）: {output_dir}")
        sys.exit(1)

    # 1. 扫描全部样本目录
    subject_dirs = []
    
    # 尝试扁平结构
    flat_shapes_dir = dataset_root / "closed_shapes_meshlib"
    if flat_shapes_dir.exists():
        subject_dirs.extend([d for d in flat_shapes_dir.iterdir() if d.is_dir()])
    
    # 尝试标准 FaceScape 分段结构
    for item in dataset_root.iterdir():
        if item.is_dir() and re.match(r"^\d{3}-\d{3}$", item.name):
            sub_shapes_dir = item / "closed_shapes_meshlib"
            if sub_shapes_dir.exists():
                subject_dirs.extend([d for d in sub_shapes_dir.iterdir() if d.is_dir()])
                
    if not subject_dirs:
        logger.error(f"在数据集根目录 {dataset_root} 下未检测到任何 closed_shapes_meshlib 样本！")
        sys.exit(1)
        
    subject_dirs = sorted(subject_dirs, key=lambda x: int(x.name) if x.name.isdigit() else 999)

    # 2. 收集并扁平化待处理的法线转换任务
    tasks = []
    for s_dir in subject_dirs:
        subject_id = s_dir.name
        closed_shapes_dir, _ = get_subject_paths(dataset_root, subject_id)
        if not closed_shapes_dir.exists():
            continue
        ply_files = list(closed_shapes_dir.glob("*.ply"))
        for mesh_path in ply_files:
            tasks.append((subject_id, mesh_path.stem, mesh_path))

    if args.instances is not None:
        if os.path.exists(args.instances):
            with open(args.instances, "r", encoding="utf-8") as f:
                instances = {line.strip() for line in f if line.strip()}
        else:
            instances = {item.strip() for item in args.instances.split(",") if item.strip()}
        tasks = [
            (subject_id, expr_name, mesh_path)
            for subject_id, expr_name, mesh_path in tasks
            if get_file_sha256(mesh_path) in instances
        ]

    logger.info(f"在目录中扫描到 {len(tasks)} 个表情样本任务，开始进行阶段二法线转换...")

    # 3. 使用 tqdm 显示进度条并执行转换，捕获失败样本信息
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    processed_samples = 0
    total_views_processed = 0
    failures = []
    records = []

    # 构造任务参数包
    tasks_args = [
        (subject_id, expr_name, mesh_path, dataset_root, output_dir, args.target_size, threads_per_worker, args.format, args.webp_quality)
        for subject_id, expr_name, mesh_path in tasks
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_subject_normals_worker, item): (item[0], item[1])
            for item in tasks_args
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="转换法线图像中"):
            subject_id, expr_name = futures[future]
            try:
                res = future.result()
                if res["success"]:
                    views_cnt = res["views_cnt"]
                    if views_cnt > 0:
                        processed_samples += 1
                        total_views_processed += views_cnt
                        records.append({
                            "sha256": get_file_sha256(get_subject_paths(dataset_root, subject_id)[0] / f"{expr_name}.ply"),
                            "cond_rendered": True,
                        })
                    else:
                        records.append({
                            "sha256": get_file_sha256(get_subject_paths(dataset_root, subject_id)[0] / f"{expr_name}.ply"),
                            "cond_rendered": False,
                        })
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

    # 4. 统一输出汇总报告与失败日志
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{minutes} 分 {seconds} 秒" if minutes > 0 else f"{seconds} 秒"

    logger.info(f"阶段二预处理圆满结束！共成功处理 {processed_samples} 个表情样本，转换法线图共计 {total_views_processed} 张，总耗时 {time_str}。")
    
    if failures:
        logger.error(f"====== 共检测到 {len(failures)} 个表情样本处理失败 ======")
        for fail in failures:
            logger.error(f"样本 #{fail['subject_id']} 表情 {fail['expr_name']} 失败原因:")
            logger.error(fail["error"])
            logger.error("=" * 60)
        logger.error("==================================================")
    else:
        logger.info("所有表情样本的法线图像全部处理成功，无失败记录。")

    if records:
        pd.DataFrame.from_records(records).to_csv(output_dir / "cond_rendered_0.csv", index=False)
        logger.info(f"条件法线转换状态记录已保存至 {output_dir / 'cond_rendered_0.csv'}")


if __name__ == "__main__":
    main()
