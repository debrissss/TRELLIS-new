import os
import shutil
import argparse
import glob
import uuid
import json
import yaml
from typing import *
import torch
import numpy as np
import imageio
import trimesh
from easydict import EasyDict as edict
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# 全局 pipeline 对象（模拟原 app.py）
pipeline = None


class MockRequest:
    """伪装的 Gradio Request 对象，用于传递 session_hash"""
    def __init__(self, session_hash: str):
        self.session_hash = session_hash


def start_session(req: MockRequest):
    """
    初始化用户会话，创建临时目录。
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: MockRequest):
    """
    结束用户会话，清理临时目录。
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """预处理输入图片"""
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """批量预处理多张输入图片"""
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    """打包状态"""
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    """还原状态"""
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """获取随机种子"""
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: MockRequest,
) -> Tuple[dict, str]:
    """核心推理"""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if not is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        outputs = pipeline.run_multi_image(
            [image[0] for image in multiimages],
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )
    # 渲染预览视频
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: MockRequest,
) -> Tuple[str, str]:
    """提取 GLB"""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(state: dict, req: MockRequest) -> Tuple[str, str]:
    """提取 Gaussian PLY"""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def extract_mesh(state: dict, req: MockRequest) -> str:
    """提取纯几何 Mesh (无纹理)"""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    _, mesh_data = unpack_state(state)
    vertices = mesh_data.vertices.cpu().numpy()
    faces = mesh_data.faces.cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh_path = os.path.join(user_dir, 'sample_mesh.ply')
    mesh.export(mesh_path)
    return mesh_path


def main():
    # 1. 预解析配置文件路径
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    args, remaining_argv = conf_parser.parse_known_args()

    # 2. 读取配置文件内容
    config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
        except Exception as e:
            print(f"Warning: 无法读取配置文件 {args.config}: {e}")

    # 3. 创建主解析器，并以配置文件作为默认值
    parser = argparse.ArgumentParser(description="TRELLIS Image to 3D CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--input_dir", type=str, default=config.get("input_dir"), help="输入图片文件夹路径")
    parser.add_argument("--output_dir", type=str, default=config.get("output_dir"), help="输出产物文件夹路径")
    parser.add_argument("--seed", type=int, default=config.get("seed", 0), help="随机种子")
    parser.add_argument("--randomize_seed", action="store_true", default=config.get("randomize_seed", False), help="是否随机化种子")
    parser.add_argument("--ss_guidance_strength", type=float, default=config.get("ss_guidance_strength", 7.5), help="稀疏结构生成 CFG 强度")
    parser.add_argument("--ss_sampling_steps", type=int, default=config.get("ss_sampling_steps", 12), help="稀疏结构生成采样步数")
    parser.add_argument("--slat_guidance_strength", type=float, default=config.get("slat_guidance_strength", 3.0), help="结构化潜变量生成 CFG 强度")
    parser.add_argument("--slat_sampling_steps", type=int, default=config.get("slat_sampling_steps", 12), help="结构化潜变量生成采样步数")
    parser.add_argument("--multiimage_algo", type=str, choices=["stochastic", "multidiffusion"], default=config.get("multiimage_algo", "stochastic"), help="多图生成算法")
    parser.add_argument("--mesh_simplify", type=float, default=config.get("mesh_simplify", 0.95), help="网格简化系数")
    parser.add_argument("--texture_size", type=int, default=config.get("texture_size", 1024), help="纹理贴图分辨率")
    
    # 重新解析所有参数，命令行输入的参数会覆盖默认值（配置文件值）
    args = parser.parse_args(remaining_argv)

    # 必填项校验（如果配置文件里也没有，则报错）
    if not args.input_dir or not args.output_dir:
        parser.print_help()
        print("\nError: 必须通过配置文件或命令行指定 --input_dir 和 --output_dir")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 扫描输入目录
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    image_paths.extend(glob.glob(os.path.join(args.input_dir, "*." + ext.split(".")[-1].upper())) for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"] if glob.glob(os.path.join(args.input_dir, "*." + ext.split(".")[-1].upper())))
    
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"):
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not image_paths:
        print(f"Error: 在 {args.input_dir} 中未找到图片。")
        return

    print(f"找到 {len(image_paths)} 张图片。")

    # 全局 pipeline 初始化
    global pipeline
    print("加载预训练模型...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "weights", "TRELLIS-image-large")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()

    is_multiimage = len(image_paths) > 1
    image_prompt = None
    multiimage_prompt = []

    if not is_multiimage:
        print("模式：单图模式")
        raw_image = Image.open(image_paths[0])
        image_prompt = preprocess_image(raw_image)
    else:
        print(f"模式：多图模式 ({args.multiimage_algo})")
        for img_path in image_paths:
            raw_image = Image.open(img_path)
            # 格式为 [(PIL.Image, path), ...]
            multiimage_prompt.append((raw_image, img_path))
        multiimage_prompt_processed = preprocess_images(multiimage_prompt)
        # 将处理后的图片替换回去，因为 image_to_3d 需要访问 image[0]
        multiimage_prompt = [(img, path) for img, (_, path) in zip(multiimage_prompt_processed, multiimage_prompt)]

    # 准备 Mock 请求和临时目录
    req = MockRequest(session_hash=uuid.uuid4().hex)
    start_session(req)

    final_seed = get_seed(args.randomize_seed, args.seed)
    print(f"使用随机种子: {final_seed}")

    try:
        print("开始生成 3D 资产...")
        state, video_path = image_to_3d(
            image=image_prompt,
            multiimages=multiimage_prompt,
            is_multiimage=is_multiimage,
            seed=final_seed,
            ss_guidance_strength=args.ss_guidance_strength,
            ss_sampling_steps=args.ss_sampling_steps,
            slat_guidance_strength=args.slat_guidance_strength,
            slat_sampling_steps=args.slat_sampling_steps,
            multiimage_algo=args.multiimage_algo,
            req=req,
        )

        print("提取 GLB 文件...")
        glb_path, _ = extract_glb(state, args.mesh_simplify, args.texture_size, req)

        print("提取 Gaussian PLY 文件...")
        gaussian_path, _ = extract_gaussian(state, req)

        print("提取纯几何 Mesh PLY 文件...")
        mesh_only_path = extract_mesh(state, req)

        # 移动文件到输出目录
        print(f"保存产物到 {args.output_dir}...")
        if os.path.exists(video_path):
            shutil.move(video_path, os.path.join(args.output_dir, "sample.mp4"))
        if os.path.exists(glb_path):
            shutil.move(glb_path, os.path.join(args.output_dir, "sample.glb"))
        if os.path.exists(gaussian_path):
            shutil.move(gaussian_path, os.path.join(args.output_dir, "sample.ply"))
        if os.path.exists(mesh_only_path):
            shutil.move(mesh_only_path, os.path.join(args.output_dir, "sample_mesh.ply"))

        print("完成！")

    finally:
        # 清理
        end_session(req)


if __name__ == "__main__":
    main()
