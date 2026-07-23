"""
数据集元数据构建与整合工具 (Dataset Metadata Build & Merge Tool)。

该脚本用于汇总、合并以及增量更新数据集的 Metadata。它支持：
1. 动态加载特定数据集的解析模块 (importlib)。
2. 合并多进程/分布式处理所产生的各类 CSV 阶段性记录文件（下载状态、渲染状态、美学评分、体素化状态、条件渲染状态、多模型特征提取等）。
3. 支持通过 Multithreading (多线程 ThreadPoolExecutor) 扫描物理磁盘文件来恢复/构建缺失的元数据标记。
4. 计算数据集资产统计指标并输出到 statistics.txt，同时打印在控制台。
"""

import os
import shutil
import sys
import time
import importlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
import utils3d

def get_first_directory(path):  
    """
    获取指定路径下的第一个子目录名称。

    Args:
        path (str): 目标检索的父目录路径。

    Returns:
        str | None: 找到的第一个子目录名称；如果未找到任何子目录，则返回 None。
    """
    with os.scandir(path) as it:  
        for entry in it:  
            if entry.is_dir():  
                return entry.name  
    return None

def need_process(key):
    """
    判断给定的元数据字段是否需要进行处理。

    Args:
        key (str): 待评估的元数据字段名称。

    Returns:
        bool: 如果该字段在待处理列表 (field) 中，或者待处理列表为 ['all']，则返回 True；否则返回 False。
    """
    return key in opt.field or opt.field == ['all']

def is_render_complete(render_dir):
    """
    检查 renders/<sha256>/ 是否包含完整渲染产物。

    完整条件：
    1. transforms.json 或 transform.json 存在；
    2. mesh.ply 存在；
    3. 000.png 到 149.png 共 150 张图片全部存在。
    """
    if not os.path.isdir(render_dir):
        return False

    has_transforms = (
        os.path.exists(os.path.join(render_dir, 'transforms.json')) or
        os.path.exists(os.path.join(render_dir, 'transform.json'))
    )
    if not has_transforms:
        return False
    if not os.path.exists(os.path.join(render_dir, 'mesh.ply')):
        return False

    return all(
        os.path.exists(os.path.join(render_dir, f'{idx:03d}.png'))
        for idx in range(150)
    )

if __name__ == '__main__':
    # 动态加载对应数据集的工具模块（第一个命令行参数指定数据集名称）
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    # 初始化命令行参数解析器 (ArgumentParser)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--field', type=str, default='all',
                        help='Fields to process, separated by commas')
    parser.add_argument('--from_file', action='store_true',
                        help='Build metadata from file instead of from records of processings.' +
                             'Useful when some processing fail to generate records but file already exists.')
    # 注入数据集特有的命令行参数
    dataset_utils.add_args(parser)
    
    # 解析命令行参数并将其转换为 EasyDict 以便以属性方式安全访问
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # 递归创建输出目录以及用于存放已合并记录备份的 merged_records 目录
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'merged_records'), exist_ok=True)

    # 将逗号分隔的字段名称解析为字段列表
    opt.field = opt.field.split(',')
    
    # 获取当前 UNIX 时间戳，用于对合并备份文件进行唯一命名
    timestamp = str(int(time.time()))

    # 加载或检索基础元数据
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        print('Loading previous metadata...')
        # 如果已存在 metadata.csv，则直接载入作为初始 Pandas DataFrame
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        # 否则通过数据集模块提供的 get_metadata 方法获取基础 DataFrame
        metadata = dataset_utils.get_metadata(**opt)
    # 将哈希值字段 sha256 设为 DataFrame 的索引列，方便进行高效的 Key-Value 更新
    metadata.set_index('sha256', inplace=True)
    
    # 合并已下载资产的记录 (downloaded_*.csv)
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('downloaded_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            # 批量读取零碎的下载记录 CSV 文件
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        # 连接所有 CSV 记录并使用 Pandas DataFrame 的 update 或 join 进行增量合并
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        if 'local_path' in metadata.columns:
            metadata.update(df, overwrite=True)
        else:
            metadata = metadata.join(df, on='sha256', how='left')
        # 将已合并的 CSV 临时记录备份移动至 merged_records 文件夹
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
            
    # 自动探测特征提取和潜空间编码 (Latent) 的模型名称
    image_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'features')):
        image_models = os.listdir(os.path.join(opt.output_dir, 'features'))
    latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'latents')):
        latent_models = os.listdir(os.path.join(opt.output_dir, 'latents'))
    ss_latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'ss_latents')):
        ss_latent_models = os.listdir(os.path.join(opt.output_dir, 'ss_latents'))
    print(f'Image models: {image_models}')
    print(f'Latent models: {latent_models}')
    print(f'Sparse Structure latent models: {ss_latent_models}')

    # 为各个处理状态字段初始化默认值（若字段不存在于 DataFrame 列中）
    if 'rendered' not in metadata.columns:
        metadata['rendered'] = [False] * len(metadata)
    if 'voxelized' not in metadata.columns:
        metadata['voxelized'] = [False] * len(metadata)
    if 'num_voxels' not in metadata.columns:
        metadata['num_voxels'] = [0] * len(metadata)
    if 'cond_rendered' not in metadata.columns:
        metadata['cond_rendered'] = [False] * len(metadata)
    # 为探测到的图像特征模型创建 Feature Column
    for model in image_models:
        if f'feature_{model}' not in metadata.columns:
            metadata[f'feature_{model}'] = [False] * len(metadata)
    # 为探测到的 Latent 模型创建 Latent Column
    for model in latent_models:
        if f'latent_{model}' not in metadata.columns:
            metadata[f'latent_{model}'] = [False] * len(metadata)
    # 为探测到的稀疏结构 Latent 模型创建 Column
    for model in ss_latent_models:
        if f'ss_latent_{model}' not in metadata.columns:
            metadata[f'ss_latent_{model}'] = [False] * len(metadata)
    
    # 合并渲染状态记录 (rendered_*.csv)
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('rendered_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # 合并美学评分记录 (aesthetic_scores_*.csv)
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('aesthetic_scores_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # 合并体素化状态记录 (voxelized_*.csv)
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('voxelized_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # 合并条件图像渲染记录 (cond_rendered_*.csv)
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('cond_rendered_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # 合并多模型特征提取记录 (feature_<model>_*.csv)
    for model in image_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'feature_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
                
    # 合并 Latent 提取记录 (latent_<model>_*.csv)
    for model in latent_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'latent_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
                
    # 合并稀疏结构 Latent 提取记录 (ss_latent_<model>_*.csv)
    for model in ss_latent_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'ss_latent_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))

    # 通过直接扫描物理磁盘文件来补全/校对元数据 (Build metadata from files)
    if opt.from_file:
        # 使用 ThreadPoolExecutor 并结合系统 CPU Count 开启多线程加速扫描过程
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(metadata), desc="Building metadata") as pbar:
            def worker(sha256):
                try:
                    # 检查渲染产物是否完整：transforms/transform、mesh.ply、000.png 到 149.png
                    if need_process('rendered'):
                        metadata.loc[sha256, 'rendered'] = is_render_complete(
                            os.path.join(opt.output_dir, 'renders', sha256)
                        )
                    # 检查体素化生成的 PLY 模型文件是否存在，并通过 utils3d 读取计算点数
                    if need_process('voxelized') and metadata.loc[sha256, 'rendered'] == True and metadata.loc[sha256, 'voxelized'] == False and \
                        os.path.exists(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')):
                        try:
                            pts = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
                            metadata.loc[sha256, 'voxelized'] = True
                            metadata.loc[sha256, 'num_voxels'] = len(pts)
                        except Exception as e:
                            pass
                    # 检查条件渲染视图配置文件是否存在
                    if need_process('cond_rendered') and metadata.loc[sha256, 'cond_rendered'] == False and \
                        os.path.exists(os.path.join(opt.output_dir, 'renders_cond', sha256, 'transforms.json')):
                        metadata.loc[sha256, 'cond_rendered'] = True
                    # 检查多模型特征提取的 NPZ 压缩文件是否存在于磁盘
                    for model in image_models:
                        if need_process(f'feature_{model}') and \
                            metadata.loc[sha256, f'feature_{model}'] == False and \
                            metadata.loc[sha256, 'rendered'] == True and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'features', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'feature_{model}'] = True
                    # 检查 Latent 空间编码提取的 NPZ 压缩文件是否存在于磁盘
                    for model in latent_models:
                        if need_process(f'latent_{model}') and \
                            metadata.loc[sha256, f'latent_{model}'] == False and \
                            metadata.loc[sha256, 'rendered'] == True and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'latents', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'latent_{model}'] = True
                    # 检查稀疏结构 Latent 提取的 NPZ 压缩文件是否存在于磁盘
                    for model in ss_latent_models:
                        if need_process(f'ss_latent_{model}') and \
                            metadata.loc[sha256, f'ss_latent_{model}'] == False and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'ss_latents', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'ss_latent_{model}'] = True
                    pbar.update()
                except Exception as e:
                    print(f'Error processing {sha256}: {e}')
                    pbar.update()
            
            # 使用 executor 将 worker 映射到所有 sha256 索引上并等待完成
            executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    # 导出整合完毕的元数据到最终的 metadata.csv
    metadata.to_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    # 计算统计指标
    num_downloaded = metadata['local_path'].count() if 'local_path' in metadata.columns else 0
    with open(os.path.join(opt.output_dir, 'statistics.txt'), 'w') as f:
        f.write('Statistics:\n')
        f.write(f'  - Number of assets: {len(metadata)}\n')
        f.write(f'  - Number of assets downloaded: {num_downloaded}\n')
        f.write(f'  - Number of assets rendered: {metadata["rendered"].sum()}\n')
        f.write(f'  - Number of assets voxelized: {metadata["voxelized"].sum()}\n')
        if len(image_models) != 0:
            f.write(f'  - Number of assets with image features extracted:\n')
            for model in image_models:
                f.write(f'    - {model}: {metadata[f"feature_{model}"].sum()}\n')
        if len(latent_models) != 0:
            f.write(f'  - Number of assets with latents extracted:\n')
            for model in latent_models:
                f.write(f'    - {model}: {metadata[f"latent_{model}"].sum()}\n')
        if len(ss_latent_models) != 0:
            f.write(f'  - Number of assets with sparse structure latents extracted:\n')
            for model in ss_latent_models:
                f.write(f'    - {model}: {metadata[f"ss_latent_{model}"].sum()}\n')
        f.write(f'  - Number of assets with captions: {metadata["captions"].count()}\n')
        f.write(f'  - Number of assets with image conditions: {metadata["cond_rendered"].sum()}\n')
        
    # 读取并打印统计指标摘要到控制台
    with open(os.path.join(opt.output_dir, 'statistics.txt'), 'r') as f:
        print(f.read())
