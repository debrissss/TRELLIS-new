"""
FaceScape 数据集元数据整合工具 (metadata.csv 生成器)。

本脚本扫描阶段一与阶段二处理生成的物理文件，计算对应 Mesh 的 SHA-256 哈希值，
自动构建并增量合并生成 TRELLIS 官方工具集可消费的 metadata.csv 文件。
"""

import os
import sys
import json
import argparse
import re
import traceback
from pathlib import Path
import pandas as pd

# 将项目根目录和当前目录加入 python 模块搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from fine_tuning.utils.logger import get_logger
from fine_tuning.utils.facescape_utils import get_facescape_subfolder, get_subject_paths, get_file_sha256

# 统一获取日志实例
logger = get_logger("build_facescape_metadata")


def main() -> None:
    """主入口，扫描物理路径并构建/更新 metadata.csv。"""
    import time
    start_time = time.time()

    parser = argparse.ArgumentParser(description="自动构建/更新 FaceScape 适配的 metadata.csv 元数据表")
    
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="FaceScape 原始数据集的根目录路径（用于定位 PLY 文件）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出和保存 metadata.csv 的目标目录")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    if not dataset_root.exists():
        logger.error(f"数据集根目录不存在: {dataset_root}")
        sys.exit(1)
        
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"

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

    # 2. 收集并扁平化待处理的任务列表以显示进度条
    tasks = []
    for s_dir in subject_dirs:
        subject_id = s_dir.name
        closed_shapes_dir, _ = get_subject_paths(dataset_root, subject_id)
        if not closed_shapes_dir.exists():
            continue
        ply_files = list(closed_shapes_dir.glob("*.ply"))
        for mesh_path in ply_files:
            tasks.append((subject_id, mesh_path.stem, mesh_path))

    logger.info(f"扫描完成，共探测到 {len(tasks)} 个表情网格模型。开始扫描物理输出并构建元数据...")

    # 3. 读取已有的 CSV (支持增量合并)
    if csv_path.exists():
        logger.info(f"检测到已存在的元数据表: {csv_path}，将进行增量更新...")
        try:
            metadata_df = pd.read_csv(csv_path)
            # 保证主键唯一并设置索引
            if "sha256" in metadata_df.columns:
                metadata_df.set_index("sha256", inplace=True)
            else:
                logger.warning("已存在的 CSV 缺失 'sha256' 字段，将重新覆盖创建。")
                metadata_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"加载已有 CSV 失败: {e}，将重新覆盖创建。")
            metadata_df = pd.DataFrame()
    else:
        metadata_df = pd.DataFrame()

    # 4. 多线程/带进度条扫描渲染文件夹
    from tqdm import tqdm
    new_records = []
    failures = []

    for subject_id, expr_name, mesh_path in tqdm(tasks, desc="扫描已处理样本"):
        try:
            sha256 = get_file_sha256(mesh_path)
            
            # 检测阶段一文件是否存在
            renders_path = output_dir / "renders" / sha256 / "transforms.json"
            rendered = renders_path.exists()
            
            # 检测阶段二文件是否存在
            cond_path = output_dir / "renders_cond" / sha256 / "transforms.json"
            cond_rendered = cond_path.exists()
            
            # 如果任何一个阶段未生成，我们可以在元数据中记录为 False，但不忽略它
            # local_path 字段通常指定原始 Mesh 的相对路径
            rel_mesh_path = str(mesh_path.relative_to(dataset_root))
            
            record = {
                "sha256": sha256,
                "local_path": rel_mesh_path,
                "rendered": rendered,
                "voxelized": False,
                "num_voxels": 0,
                "cond_rendered": cond_rendered,
                "captions": f"A standard face expression mesh of subject {subject_id}, expression {expr_name}"
            }
            new_records.append(record)
        except Exception as e:
            failures.append({
                "subject_id": subject_id,
                "expr_name": expr_name,
                "error": f"{str(e)}\n{traceback.format_exc()}"
            })

    if new_records:
        # 将新扫描数据转为 DataFrame
        scan_df = pd.DataFrame(new_records)
        scan_df.set_index("sha256", inplace=True)
        
        # 与旧数据进行合并 (覆盖已有列状态，合并新行)
        if not metadata_df.empty:
            # 合并前更新冲突的列
            metadata_df = metadata_df.combine_first(scan_df)
            metadata_df.update(scan_df, overwrite=True)
        else:
            metadata_df = scan_df
            
        # 写入磁盘
        metadata_df.to_csv(csv_path)
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes} 分 {seconds} 秒" if minutes > 0 else f"{seconds} 秒"

        logger.info(f"元数据 metadata.csv 已成功保存至 {csv_path}。当前共包含 {len(metadata_df)} 个记录，本步骤总耗时 {time_str}。")
    else:
        logger.warning("未扫描到任何有效数据记录，未更新 CSV。")

    # 错误统计
    if failures:
        logger.error(f"====== 扫描过程中检测到 {len(failures)} 个失败样本 ======")
        for fail in failures:
            logger.error(f"样本 #{fail['subject_id']} 表情 {fail['expr_name']} 扫描失败:")
            logger.error(fail["error"])
            logger.error("=" * 60)
    else:
        logger.info("扫描物理文件与构建 CSV 过程无任何异常。")


if __name__ == "__main__":
    main()
