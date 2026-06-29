import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d


def _voxelize(file, sha256, output_dir):
    """对三维网格模型进行体素化处理。

    Args:
        file (str): 网格文件的路径或标识（在此函数中未使用，由外层调度传入）。
        sha256 (str): 模型的 SHA256 哈希值，用于定位渲染结果和保存体素模型。
        output_dir (str): 数据集处理的输出根目录。

    Returns:
        dict: 包含处理状态的字典：
            - 'sha256' (str): 模型的 SHA256 哈希值。
            - 'voxelized' (bool): 是否成功体素化（True）。
            - 'num_voxels' (int): 体素化后的体素网格顶点数量。
    """
    # 读取渲染出来的网格模型
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, 'mesh.ply'))
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # 将三维网格模型在其边界范围内转化为体素大小为 1/64 的 VoxelGrid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    # 提取所有体素的网格坐标索引
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    # 验证体素索引是否在合法范围 [0, 63] 内
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    # 将体素整数索引转换回 [-0.5, 0.5] 空间中的体素中心坐标
    vertices = (vertices + 0.5) / 64 - 0.5
    # 将计算得到的体素坐标保存为 ply 文件
    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), vertices)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

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
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'voxels'), exist_ok=True)

    # get file list
    # 检查并加载数据集的元数据 CSV 文件
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        # 如果未指定特定实例列表，则进行常规过滤（过滤低美学评分、未渲染模型和已体素化模型）
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == True]
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
        # 如果指定了特定实例列表，则只处理指定的实例（支持从文件加载或以逗号分隔的字符串解析）
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

    # filter out objects that are already processed
    # 检查本地磁盘是否已存在已生成的体素 ply 文件，以实现增量处理（断点续传）
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')):
            pts = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
            records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    # 使用偏函数绑定输出目录，并并行/顺序地对所有实例执行体素化处理，最后保存元数据
    func = partial(_voxelize, output_dir=opt.output_dir)
    voxelized = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)
