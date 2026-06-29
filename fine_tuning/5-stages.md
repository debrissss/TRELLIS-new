针对你的 3D 人脸微调数据结构，以下是整合了 **视角过滤**、**条件法线提取** 与 **Blender 多视角图像自动渲染** 后的完整 3D 人脸数据预处理管线。

我们将整个管线划分为以下核心处理阶段，并详细说明每个阶段的输入、输出格式与运行指令：

---

### 阶段一：Mesh 与相机的规范化缩放（Scale & Pose Normalization）
由于数据集内所有人脸均已对齐，为保持人脸的大小比例关系，我们将对所有样本应用**固定的缩放系数 $S$** 和**位移向量 $T$**，使其顶点均落入 $[-0.5, 0.5]^3$ 空间。为了不破坏相机的重投影一致性，我们将同步修正相机的外参平移向量，实现“无感知”的坐标系变换。

*   **输入形式**：
    *   原始 PLY 格式面部 Mesh 路径：`facescape/001-020/{id}/closed_shapes_meshlib/{id}/1_neutral.ply`
    *   相机参数 JSON 文件：`facescape/001-020/{id}/aligned_camera_params/{id}/1_neutral/params.json`
*   **处理内容**：
    1.  **网格坐标变换**：将网格顶点坐标通过 $X' = S \cdot X + T$ 进行变换（$S$ 和 $T$ 对全部数据固定）。
    2.  **相机外参变换**：外参旋转矩阵 $R$ 保持不变；平移向量修正为：$t' = S \cdot t - R \cdot T$。内参矩阵 $K$ 保持不变。这确保了在缩放后的 3D 坐标空间下，重投影到 2D 像平面上的像素位置与变换前完全相同，避免了后续投影特征提取出现偏差。
    3.  **视角过滤**：丢弃无效视角（偏航角与俯仰角超限），仅保留有效视角的相机参数。
*   **输出形式**：
    *   缩放后的三角网格：保存为 `<OUTPUT_DIR>/renders/{sha256}/mesh.ply`
    *   更新后的相机外参：以有效视角索引为 Key，保存为 `<OUTPUT_DIR>/renders/{sha256}/transforms.json`
*   **运行命令**：
    ```bash
    python fine_tuning/preprocess_stage1.py --dataset_root <FaceScape原始数据集路径> --output_dir <规范化数据集输出路径>
    ```

---

### 阶段二：无掩码法线图的 4 通道 RGBA 融合（Normal Map Post-Processing）
针对法线图没有 Mask 的问题，在数学上可以通过阈值提取前背景。因为法线图的 RGB 对应单位法向量的映射值 $[(n_x+1)/2, (n_y+1)/2, (n_z+1)/2] \times 255$。因为单位法向量的模长必须为 1，所以合法的法线像素不可能三通道同时达到接近 255 的极值。因此，任何有效物体的法线像素值都远远偏离纯白色，我们可以安全地利用纯白背景提取高精度的 Mask。

*   **输入形式**：
    *   原始多视角法线图路径：`facescape/001-020/{id}/normals/{id}/1_neutral/{i}.png`
    *   有效视角索引列表。
*   **处理内容**：
    1.  读取有效视角的法线图。
    2.  提取 Alpha 通道（Mask）：设定一个高阈值（如 $\text{RGB} > 254$），凡是接近纯白的背景区域 Alpha 设为 `0`（透明），其余人脸区域 Alpha 设为 `255`（不透明）。
    3.  将 3 通道法线 RGB 图与 1 通道 Alpha 图合并，生成 4 通道 RGBA PNG 格式图像。
    4.  生成对应的条件视图索引清单 `transforms.json`。
*   **输出形式**：
    *   4 通道条件图：保存为 `<OUTPUT_DIR>/renders_cond/{sha256}/normal_{i}.png`
    *   条件视图索引：保存为 `<OUTPUT_DIR>/renders_cond/{sha256}/transforms.json`（仅列出有效视角的 `file_path`）
*   **运行命令**：
    ```bash
    python fine_tuning/preprocess_stage2.py --dataset_root <FaceScape原始数据集路径> --output_dir <规范化数据集输出路径> --num_workers 8
    ```

---

### 阶段三：三维人脸体素化（Voxelization）
*   **输入形式**：
    *   阶段一导出的规范三角网格：`<OUTPUT_DIR>/renders/{sha256}/mesh.ply`
*   **处理内容**：
    *   调用 Open3D 库，对缩放后的 mesh 进行体素化（网格步长为 $1/64$），提取被占用的体素中心点云。
*   **输出形式**：
    *   体素化点云：保存为 `<OUTPUT_DIR>/voxels/{sha256}.ply`（空间范围限制在 $[-0.5, 0.5]^3$ 内）
*   **运行命令** (使用带有安全硬超时保护和自动断点续传的自定义脚本)：
    ```bash
    python fine_tuning/voxelize.py Toys4k --output_dir <规范化数据集输出路径> --max_workers 8 --timeout 5.0
    ```

---

### 阶段三点五：多视角图像渲染 (Multi-view Image Rendering)
原版 TRELLIS 在特征提取阶段需要 150 个由 Blender 离线渲染的多视角法线图（用来重投影赋予 3D 体素丰富的密集 2D 特征）。这一步在特征提取前执行，仅进行图像渲染，不重新保存/覆盖已对齐的网格。

*   **输入形式**：
    *   阶段一导出的规范三角网格：`<OUTPUT_DIR>/renders/{sha256}/mesh.ply`
*   **处理内容**：
    *   自动调用 Blender 环境，以规范化网格为渲染对象，在球面均匀采样并渲染 150 个视角的 `512x512` 法线图。
    *   自动生成并覆盖 `renders/{sha256}/transforms.json`，其中包含这 150 个虚拟相机的位置和内外参信息。
*   **输出形式**：
    *   150 张视角图像：保存为 `<OUTPUT_DIR>/renders/{sha256}/{000-149}.png`
    *   150 个视角对应的相机矩阵：保存为 `<OUTPUT_DIR>/renders/{sha256}/transforms.json`
*   **运行命令**：
    ```bash
    python fine_tuning/render.py Toys4k --output_dir <规范化数据集输出路径> --max_workers 8
    ```

---

### 阶段四：图像特征重投影融合（Feature Projection & Fusion）
此步骤需要将 2D 语义先验投影赋予体素。为了保证目标潜空间变量的高质量，此阶段加载阶段三点五中渲染出的 150 个视角图，用 DINOv2 提取特征图并反投影赋予 3D 体素。

*   **输入形式**：
    *   体素点云：`<OUTPUT_DIR>/voxels/{sha256}.ply`
    *   阶段三点五中 150 个视角导出的相机矩阵与渲染图：`<OUTPUT_DIR>/renders/{sha256}/` 目录
*   **处理内容**：
    1.  调用 DINOv2 提取 150 张视角图的密集局部 Token 描述符。
    2.  利用相机参数，将 3D 中的体素点重投影到这 150 个相机的像平面，并双线性采样 2D 特征。
    3.  取多视角投影均值，融合为每个体素点的 1024 维语义特征。
*   **输出形式**：
    *   融合特征文件：保存为 `<OUTPUT_DIR>/features/dinov2_vitl14_reg/{sha256}.npz`
*   **运行命令** (使用带有死锁防护和路径弹性兼容的优化版脚本)：
    ```bash
    python fine_tuning/extract_feature.py --output_dir <规范化数据集输出路径> --batch_size 16
    ```

---

### 阶段五：SLaT 潜编码转换（SLat Latent Encoding）
*   **输入形式**：
    *   特征文件：`<OUTPUT_DIR>/features/dinov2_vitl14_reg/{sha256}.npz`
*   **处理内容**：
    *   加载特征 NPZ 构建 `SparseTensor`，送入预训练的 3D 稀疏 Swin-Transformer VAE 编码器（`slat_enc_swin8`）提取最终的潜变量。
*   **输出形式**：
    *   结构化目标潜编码（$x_0$）：保存为 `<OUTPUT_DIR>/latents/{latent_model}/{sha256}.npz`
*   **运行命令**：
    ```bash
    python dataset_toolkits/encode_latent.py --output_dir <规范化数据集输出路径>
    python dataset_toolkits/encode_ss_latent.py --output_dir <规范化数据集输出路径>
    ```

---

### 阶段五点五：生成元数据汇总与统计数据（Build Final Metadata & Stats）
*   **运行命令**：
    ```bash
    python dataset_toolkits/stat_latent.py --output_dir <规范化数据集输出路径>
    python fine_tuning/build_facescape_metadata.py --dataset_root <FaceScape原始数据集路径> --output_dir <规范化数据集输出路径>
    ```