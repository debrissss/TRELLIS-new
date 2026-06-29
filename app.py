"""
TRELLIS Gradio 服务端应用。

本模块实现了基于 Gradio 的 Web 界面，用于将单张或多张图片转换为 3D 资产（包括 Gaussian Splatting 和 Mesh）。
"""

import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import shutil
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)


def start_session(req: gr.Request):
    """
    初始化用户会话，创建临时目录。

    Args:
        req (gr.Request): Gradio 请求对象，包含会话信息。
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    """
    结束用户会话，清理临时目录。

    Args:
        req (gr.Request): Gradio 请求对象，包含会话信息。
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    预处理输入图片（如背景去除、缩放等）。

    Args:
        image (Image.Image): 输入的原始图片。

    Returns:
        Image.Image: 预处理后的图片。
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """
    批量预处理多张输入图片。

    Args:
        images (List[Tuple[Image.Image, str]]): 输入图片列表，每项为 (图片, 路径) 元组。

    Returns:
        List[Image.Image]: 预处理后的图片列表。
    """
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    """
    将生成的 Gaussian 和 Mesh 结果打包为可序列化的字典，用于状态保存。

    Args:
        gs (Gaussian): Gaussian Splatting 表示结果。
        mesh (MeshExtractResult): Mesh 提取结果。

    Returns:
        dict: 包含 3D 资产数据的字典。
    """
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
    
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    """
    从状态字典中还原 Gaussian 和 Mesh 对象。

    Args:
        state (dict): 包含 3D 资产数据的状态字典。

    Returns:
        Tuple[Gaussian, edict]: 还原后的 Gaussian 对象和包含顶点/面信息的 Mesh 对象。
    """
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
    """
    获取随机种子。

    Args:
        randomize_seed (bool): 是否使用随机种子。
        seed (int): 手动指定的种子。

    Returns:
        int: 最终使用的种子值。
    """
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
    req: gr.Request,
) -> Tuple[dict, str]:
    """
    将图像转换为 3D 模型，并生成预览视频。

    Args:
        image (Image.Image): 输入的单张图像。
        multiimages (List[Tuple[Image.Image, str]]): 多图模式下的输入图像列表。
        is_multiimage (bool): 是否处于多图模式。
        seed (int): 随机种子。
        ss_guidance_strength (float): 稀疏结构生成阶段的 CFG 强度。
        ss_sampling_steps (int): 稀疏结构生成阶段的采样步数。
        slat_guidance_strength (float): 结构化潜变量生成阶段的 CFG 强度。
        slat_sampling_steps (int): 结构化潜变量生成阶段的采样步数。
        multiimage_algo (Literal["multidiffusion", "stochastic"]): 多图生成算法。
        req (gr.Request): Gradio 请求对象。

    Returns:
        Tuple[dict, str]: (打包的状态字典, 渲染视频的本地路径)。
    """
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
    req: gr.Request,
) -> Tuple[str, str]:
    """
    从生成的结果中提取 GLB 文件。

    Args:
        state (dict): 包含生成的 3D 模型数据的状态字典。
        mesh_simplify (float): 网格简化系数（0-1，越大简化越少）。
        texture_size (int): 纹理贴图分辨率。
        req (gr.Request): Gradio 请求对象。

    Returns:
        Tuple[str, str]: GLB 文件的路径（用于展示和下载）。
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    从生成的结果中提取 Gaussian PLY 文件。

    Args:
        state (dict): 包含生成的 3D 模型数据的状态字典。
        req (gr.Request): Gradio 请求对象。

    Returns:
        Tuple[str, str]: PLY 文件的路径（用于展示和下载）。
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def prepare_multi_example() -> List[Image.Image]:
    """
    准备多图模式下的示例图片。

    Returns:
        List[Image.Image]: 拼接后的示例图片列表。
    """
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    """
    将横向拼接的示例图拆分为多张独立的视角图，并进行预处理。

    Args:
        image (Image.Image): 拼接后的原始图片。

    Returns:
        List[Image.Image]: 拆分并预处理后的图片列表。
    """
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
    * Upload an image and click "Generate" to create a 3D asset. If the image has alpha channel, it be used as the mask. Otherwise, we use `rembg` to remove the background.
    * If you find the generated 3D asset satisfactory, click "Extract GLB" to extract the GLB file and download it.
    """)
    
    with gr.Row():
        # 左侧列：输入和设置
        with gr.Column():
            # 输入标签页：支持单图和多图
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="Single Image", id=0) as single_image_input_tab:
                    image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=300)
                with gr.Tab(label="Multiple Images", id=1) as multiimage_input_tab:
                    multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=300, columns=3)
                    gr.Markdown("""
                        Input different views of the object in separate images. 
                        
                        *NOTE: this is an experimental algorithm without training a specialized model. It may not produce the best results for all images, especially those having different poses or inconsistent details.*
                    """)
            
            # 生成设置折叠面板
            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="stochastic")

            generate_btn = gr.Button("Generate")
            
            # GLB 提取设置
            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
            
            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
            gr.Markdown("""
                        *NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*
                        """)

        # 右侧列：预览和输出
        with gr.Column():
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)
            
            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)  
    
    # 内部状态管理
    is_multiimage = gr.State(False)
    output_buf = gr.State()

    # Example images at the bottom of the page
    with gr.Row() as single_image_example:
        examples = gr.Examples(
            examples=[
                f'assets/example_image/{image}'
                for image in os.listdir("assets/example_image")
            ],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=64,
        )
    with gr.Row(visible=False) as multiimage_example:
        examples_multi = gr.Examples(
            examples=prepare_multi_example(),
            inputs=[image_prompt],
            fn=split_image,
            outputs=[multiimage_prompt],
            run_on_click=True,
            examples_per_page=8,
        )

    # 事件处理器定义
    demo.load(start_session)
    demo.unload(end_session)
    
    # 切换单图/多图模式
    single_image_input_tab.select(
        lambda: tuple([False, gr.Row.update(visible=True), gr.Row.update(visible=False)]),
        outputs=[is_multiimage, single_image_example, multiimage_example]
    )
    multiimage_input_tab.select(
        lambda: tuple([True, gr.Row.update(visible=False), gr.Row.update(visible=True)]),
        outputs=[is_multiimage, single_image_example, multiimage_example]
    )
    
    # 图片上传预处理
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )
    multiimage_prompt.upload(
        preprocess_images,
        inputs=[multiimage_prompt],
        outputs=[multiimage_prompt],
    )

    # 点击生成按钮
    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[image_prompt, multiimage_prompt, is_multiimage, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, multiimage_algo],
        outputs=[output_buf, video_output],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    # 视频输出清除逻辑
    video_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    # 提取 GLB 逻辑
    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_glb],
    )
    
    # 提取 Gaussian 逻辑
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    # 3D 模型输出清除逻辑
    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_glb],
    )
    

# 启动 Gradio 应用
if __name__ == "__main__":
    # 加载预训练的 TRELLIS 流水线
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    demo.launch()
