"""
体素化数据处理脚本 (FaceScape 微调优化版)
采用子进程隔离执行以支持硬超时判定，不再打印单样本运行日志以保持进度条整洁，最终汇总输出并保存跳过的坏样本。
"""

import os
# 设置线程控制环境变量，防止 Open3D 内部 OpenMP 多线程与 Python ThreadPoolExecutor 冲突导致死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import sys
import json
import argparse
import threading
import importlib
import multiprocessing
from functools import partial
import pandas as pd
from easydict import EasyDict as edict
import numpy as np

# 将项目根目录和 dataset_toolkits 目录加入 Python 模块搜索路径，确保能正确导入 datasets 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset_toolkits")))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import open3d as o3d
import utils3d

# 全局互斥锁和被跳过的坏样本记录容器
skipped_lock = threading.Lock()
skipped_samples = []


def _voxelize_worker(mesh_path, sha256, output_ply_path, conn):
    """在子进程中运行的物理体素化计算工人，完全隔离 C++ 运行环境防止死锁。

    Args:
        mesh_path (str): 原始渲染 Mesh PLY 文件的物理路径。
        sha256 (str): 该物体的 SHA-256 唯一哈希标示。
        output_ply_path (str): 输出体素 PLY 文件的保存路径。
        conn (multiprocessing.connection.Connection): 用于将结果回传给父进程的管道连接。
    """
    try:
        # 子进程环境中再次强制锁定单线程，确保在任何平台上都是线程安全的
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        import open3d as o3d
        import numpy as np
        import utils3d

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        # 物理保底判定：如果网格为空或没有顶点，直接返回失败
        if mesh.is_empty() or not mesh.has_vertices():
            conn.send({"success": False, "reason": "Mesh is empty or has no vertices"})
            return

        # 对顶点坐标进行裁剪限制在 [-0.5, 0.5] 范围内
        vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        # 在指定物理范围限制内将 Mesh 转化为 1/64 的 VoxelGrid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh, 
            voxel_size=1/64, 
            min_bound=(-0.5, -0.5, -0.5), 
            max_bound=(0.5, 0.5, 0.5)
        )
        
        voxels = voxel_grid.get_voxels()
        if not voxels:
            conn.send({"success": False, "reason": "No voxels generated from mesh geometry"})
            return

        grid_indices = np.array([voxel.grid_index for voxel in voxels])
        
        # 验证体素索引是否在合法范围 [0, 63] 内，超出则强制 clip 规避 Crash
        grid_indices = np.clip(grid_indices, 0, 63)

        # 将体素整数索引转换回 [-0.5, 0.5] 空间中的体素中心坐标
        vertices = (grid_indices + 0.5) / 64 - 0.5
        
        # 将体素中心坐标以 PLY 形式输出保存
        utils3d.io.write_ply(output_ply_path, vertices)
        conn.send({"success": True, "num_voxels": len(vertices)})
    except Exception as e:
        conn.send({"success": False, "reason": str(e)})


def _voxelize(file, sha256, output_dir, timeout=5.0):
    """对单样本进行体素化处理包装器，包含已有文件校验与多进程超时拦截逻辑。

    Args:
        file (str): 文件的路径或标识（本函数中主要通过 sha256 定位）。
        sha256 (str): 模型的 SHA256 哈希值，用于唯一标示样本。
        output_dir (str): 数据处理输出根目录。
        timeout (float): 单个样本体素化计算的最大允许秒数。

    Returns:
        dict: 包含处理状态的字典，处理失败或跳过则返回 None。
    """
    mesh_path = os.path.join(output_dir, 'renders', sha256, 'mesh.ply')
    if not os.path.exists(mesh_path):
        with skipped_lock:
            skipped_samples.append({"sha256": sha256, "reason": "Input mesh.ply not found"})
        return None

    output_ply_path = os.path.join(output_dir, 'voxels', f'{sha256}.ply')
    
    # 优化 3：检查已有体素 PLY 文件是否完好，若完好直接跳过
    if os.path.exists(output_ply_path):
        try:
            pts = utils3d.io.read_ply(output_ply_path)[0]
            # 文件正常无损坏，跳过处理直接返回数据
            return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)}
        except Exception:
            # 校验失败说明文件损坏，物理删除后重新排队生成
            try:
                os.remove(output_ply_path)
            except Exception:
                pass

    # 优化 1：使用 multiprocessing.Process 开启子进程，控制 5.0 秒超时阈值
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(
        target=_voxelize_worker, 
        args=(mesh_path, sha256, output_ply_path, child_conn)
    )
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        # 强制终止卡死的 C++ 运算进程
        p.terminate()
        p.join()
        
        # 物理删除可能写到一半的残缺不完整文件
        if os.path.exists(output_ply_path):
            try:
                os.remove(output_ply_path)
            except Exception:
                pass
        with skipped_lock:
            skipped_samples.append({"sha256": sha256, "reason": f"Voxelization timed out after {timeout} seconds"})
        return None

    # 获取子进程执行状态反馈
    if parent_conn.poll():
        res = parent_conn.recv()
        if res.get("success"):
            return {'sha256': sha256, 'voxelized': True, 'num_voxels': res["num_voxels"]}
        else:
            with skipped_lock:
                skipped_samples.append({"sha256": sha256, "reason": res.get("reason", "Unknown error")})
            return None
    else:
        with skipped_lock:
            skipped_samples.append({"sha256": sha256, "reason": "Subprocess exited unexpectedly without data"})
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fine_tuning/voxelize.py <dataset_name> [args...]")
        sys.exit(1)

    dataset_name = sys.argv[1]
    dataset_utils = importlib.import_module(f'datasets.{dataset_name}')

    parser = argparse.ArgumentParser()
    # 输出结果保存目录（包含体素化 PLY 文件与临时 CSV 记录）
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    # 美学评分过滤阈值，过滤掉低于此分数的低质量模型（默认 None）
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    # 指定需要处理的特定样本（可以是 sha256 列表或存有列表的文件，默认 None 代表全部）
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    # 渲染的相机视角数量，仅在部分数据集逻辑中用于校对，体素化并不直接依赖它
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    # 动态载入对应数据集适配器特有的命令行参数
    dataset_utils.add_args(parser)
    # 当前进程节点的 Rank 编号，用于多机分布式处理时进行任务切片分配
    parser.add_argument('--rank', type=int, default=0)
    # 分布式处理中的节点总数，配合 rank 共同实现多机任务均分
    parser.add_argument('--world_size', type=int, default=1)
    # 并行处理的最大工作线程数，限制并发数防止内存溢出（默认使用 CPU 核心数）
    parser.add_argument('--max_workers', type=int, default=None)
    # 优化点 1：控制单个样本体素化计算的最大允许秒数
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='Timeout in seconds for voxelizing each sample')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'voxels'), exist_ok=True)

    # 检查并加载数据集的元数据 CSV 文件
    metadata_csv_path = os.path.join(opt.output_dir, 'metadata.csv')
    if not os.path.exists(metadata_csv_path):
        raise ValueError(f'metadata.csv not found at {metadata_csv_path}')
    metadata = pd.read_csv(metadata_csv_path)

    if opt.instances is None:
        # 如果未指定特定实例列表，则进行常规过滤
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == True]
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
        # 如果指定了特定实例列表，则只处理指定的实例
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # 根据当前进程的 rank 和总 world_size 对数据分片，以便多进程分布式处理
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # 检查本地磁盘是否已存在已生成的体素 ply 文件，以实现增量处理（断点续传）
    # 增加 try-except 容错，如果已存在的 ply 文件损坏，自动删除并重新排队处理
    for sha256 in copy.copy(metadata['sha256'].values):
        ply_path = os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')
        if os.path.exists(ply_path):
            try:
                pts = utils3d.io.read_ply(ply_path)[0]
                records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
                metadata = metadata[metadata['sha256'] != sha256]
            except Exception as e:
                print(f"[Warning] Corrupted ply file detected at {ply_path}. Error: {e}. Removing and re-processing.")
                try:
                    os.remove(ply_path)
                except Exception:
                    pass
                
    print(f'Processing {len(metadata)} objects...')

    # 使用偏函数绑定输出目录和超时阈值，并并行地对所有实例执行体素化处理，最后保存元数据
    func = partial(_voxelize, output_dir=opt.output_dir, timeout=opt.timeout)
    voxelized = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    with skipped_lock:
        failed_records = [
            {
                'sha256': item['sha256'],
                'voxelized': False,
                'num_voxels': 0,
                'error': item['reason'],
            }
            for item in skipped_samples
        ]
    voxelized = pd.concat([
        voxelized,
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(failed_records),
    ])
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)
    
    # 优化点 2：隐藏内部单样本状态打印，在末尾统一输出跳过的坏样本总结
    print(f"\nVoxelization stage completed for rank {opt.rank}. Output metadata saved to voxelized_{opt.rank}.csv")
    
    with skipped_lock:
        total_skipped = len(skipped_samples)
        if total_skipped > 0:
            print(f"\n==================================================")
            print(f"[Summary] Skipped {total_skipped} objects during voxelization:")
            for item in skipped_samples:
                print(f"  - SHA256: {item['sha256']} | Reason: {item['reason']}")
            print(f"==================================================\n")
            
            # 保存到 skipped_voxels.json 文件以便持久化追溯
            json_path = os.path.join(opt.output_dir, f"skipped_voxels_{opt.rank}.json")
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(skipped_samples, f, indent=4, ensure_ascii=False)
                print(f"Detailed skip list saved to: {json_path}")
            except Exception as e:
                print(f"[Error] Failed to save skipped list to JSON: {e}")
        else:
            print(f"\n[Summary] All objects processed successfully without any skips!")
