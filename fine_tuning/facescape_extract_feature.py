import os
import copy
import sys
import json
import importlib
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from PIL import Image


# 禁用 PyTorch 梯度计算以节省显存并加速推理
torch.set_grad_enabled(False)


def get_data(frames, sha256):
    """从渲染的多视角图像中加载并预处理图像及相机参数。

    Args:
        frames (list): 包含各视角相机参数的帧信息列表，每个元素是字典，包含 'file_path', 'transform_matrix', 'camera_angle_x'。
        sha256 (str): 当前处理模型的 SHA256 哈希值。

    Yields:
        dict: 预处理后的单个视角数据，包含：
            - 'image' (torch.Tensor): 形状为 (3, 518, 518) 的 RGB 图像张量，已应用 alpha 通道背景乘积。
            - 'extrinsics' (torch.Tensor): 形状为 (4, 4) 的相机外参矩阵。
            - 'intrinsics' (torch.Tensor): 形状为 (3, 3) 的相机内参矩阵。
    """
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            image_path = os.path.join(opt.output_dir, 'renders', sha256, view['file_path'])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances file or comma-separated sha256 values')
    parser.add_argument('--voxel_dir', type=str, default='voxels',
                        help='Voxel directory under output_dir')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute features even if existing npz files are present')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, 'features', feature_name), exist_ok=True)

    # 加载 DINOv2 预训练模型并移动至 GPU，切换为评估模式
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()
    # 图像归一化变换，使用 ImageNet 的均值和标准差
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 518x518 图像输入 DINOv2 ViT 14x14 patch 后得到的特征图大小 (37x37)
    n_patch = 518 // 14

    # 获取待处理的元数据列表
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(
            os.path.join(opt.output_dir, 'metadata.csv'),
            dtype={'sha256': str},
        )
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = [
                item.strip() for item in opt.instances.split(',')
                if item.strip()
            ]
        metadata = metadata[metadata['sha256'].isin(instances)]
        if metadata.empty:
            raise ValueError(
                'None of the requested instances were found in metadata.csv'
            )
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if f'feature_{feature_name}' in metadata.columns:
            metadata = metadata[metadata[f'feature_{feature_name}'] == False]
        metadata = metadata[metadata['voxelized'] == True]
        metadata = metadata[metadata['rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # 过滤掉本地已生成特征 npz 文件的实例以支持断点续传
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if not opt.overwrite and os.path.exists(os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'feature_{feature_name}' : True})
            sha256s.remove(sha256)

    # 启动多线程特征提取队列逻辑
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
            ThreadPoolExecutor(max_workers=8) as saver_executor:
            def loader(sha256):
                """多线程数据加载函数，负责读取转换参数、预处理图像并读取对应的体素点云位置"""
                try:
                    with open(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    data = []
                    # 多线程并行读取并预处理每个视角的图像及相机参数
                    for datum in get_data(frames, sha256):
                        datum['image'] = transform(datum['image'])
                        data.append(datum)
                    # 读取体素化的点云顶点位置
                    positions = utils3d.io.read_ply(
                        os.path.join(
                            opt.output_dir,
                            opt.voxel_dir,
                            f'{sha256}.ply',
                        )
                    )[0]
                    # 放入阻塞队列以供主推理循环消费
                    load_queue.put((sha256, data, positions, None))
                except Exception as e:
                    print(f"Error loading data for {sha256}: {e}")
                    load_queue.put((sha256, None, None, str(e)))

            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack, patchtokens_mean):
                """多线程保存函数，将已完成多视角平均的特征写入磁盘。"""
                # 保持原始脚本的最终存储精度：多视角平均后转为 float16。
                pack['patchtokens'] = patchtokens_mean.astype(np.float16)
                # 以压缩 of npz 格式保存体素索引与提取到的特征
                save_path = os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'feature_{feature_name}' : True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting features"):
                # 从加载队列中获取已预处理的数据
                sha256, data, positions, load_error = load_queue.get()
                if load_error is not None:
                    records.append({'sha256': sha256, f'feature_{feature_name}' : False, 'error': load_error})
                    continue
                with torch.inference_mode():
                    positions = torch.from_numpy(positions).float().cuda()
                    # 将体素中心点坐标映射到 [0, 63]^3 的离散网格索引
                    indices = ((positions + 0.5) * 64).long()
                    assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
                    n_views = len(data)
                    pack = {
                        'indices': indices.cpu().numpy().astype(np.uint8),
                    }
                    patchtokens_sum = None

                    # 按批次批处理多视角图像的前向传播与相机投影。
                    # 与官方脚本一致地使用 DINO 特征、project_cv 和 grid_sample；
                    # 区别仅在于每个 batch 立即采样并累加，避免保留 150 个视角的完整特征图。
                    for i in range(0, n_views, opt.batch_size):
                        batch_data = data[i:i+opt.batch_size]
                        bs = len(batch_data)
                        batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
                        batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
                        batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
                        
                        # 提取多视角图像的 DINOv2 图像特征
                        features = dinov2_model(batch_images, is_training=True)
                        # 将 3D 点云投影至 2D 相机平面，返回裁剪后的 UV 坐标并归一化至 [-1, 1] 范围
                        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                        # 提取特征图特征（排除 DINOv2 的 class token 与 register tokens），并重构成 2D 特征图结构
                        patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                        sampled = F.grid_sample(
                            patchtokens,
                            uv.unsqueeze(1),
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(2).permute(0, 2, 1)
                        sampled_sum = sampled.sum(dim=0).cpu()
                        if patchtokens_sum is None:
                            patchtokens_sum = sampled_sum
                        else:
                            patchtokens_sum += sampled_sum

                    patchtokens_mean = (patchtokens_sum / n_views).numpy()

                # save features
                # 提交给保存线程池，进行特征异步插值与保存落盘
                saver_executor.submit(saver, sha256, pack, patchtokens_mean)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'feature_{feature_name}_{opt.rank}.csv'), index=False)
        
