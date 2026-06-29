# TRELLIS-500K

TRELLIS-500K 是一个包含 500K 个 3D 资产 (3D assets) 的数据集 (dataset)，这些资产是从 [Objaverse(XL)](https://objaverse.allenai.org/)、[ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)、[3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)、[HSSD](https://huggingface.co/datasets/hssd/hssd-models) 和 [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k) 中挑选 (curated) 出来的，并基于美学评分 (aesthetic scores) 进行了过滤。
该数据集 (dataset) 用于 3D 生成任务 (3D generation tasks)。

该数据集 (dataset) 以包含 3D 资产 (3D assets) 元数据 (metadata) 的 csv 文件 (csv files) 形式提供。

## 数据集统计信息 (Dataset Statistics)

下表总结了数据集 (dataset) 的过滤 (filtering) 和构成 (composition)：

***注意：部分 3D 资产 (3D assets) 缺少文本描述 (text captions)。如果需要描述 (captions)，请过滤掉这些资产。***
| 来源 (Source) | 美学评分阈值 (Aesthetic Score Threshold) | 过滤后大小 (Filtered Size) | 带有描述 (With Captions) |
|:-:|:-:|:-:|:-:|
| ObjaverseXL (sketchfab) | 5.5 | 168307 | 167638 |
| ObjaverseXL (github) | 5.5 | 311843 | 306790 |
| ABO | 4.5 | 4485 | 4390 |
| 3D-FUTURE | 4.5 | 9472 | 9291 |
| HSSD | 4.5 | 6670 | 6661 |
| 所有（训练集 (training set)） | - | 500777 | 494770 |
| Toys4k（评估集 (evaluation set)） | 4.5 | 3229 | 3180 |

## 数据集位置 (Dataset Location)

该数据集 (dataset) 托管 (hosted) 在 Hugging Face Datasets 上。您可以在此处预览该数据集 (dataset)：

[https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K)

无需手动下载 csv 文件 (csv files)。我们提供了用于加载和准备数据集 (dataset) 的工具包 (toolkits)。

## 数据集工具包 (Dataset Toolkits)

我们提供了用于数据准备的 [工具包 (toolkits)](dataset_toolkits)。

### 步骤 1：安装依赖项 (Install Dependencies)

```
. ./dataset_toolkits/setup.sh
```

### 步骤 2：加载元数据 (Load Metadata)

首先，我们需要加载数据集 (dataset) 的元数据 (metadata)。

```
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <OUTPUT_DIR> [--source <SOURCE>]
```

- `SUBSET`：要加载的数据集子集 (subset)。选项 (options) 包括 `ObjaverseXL`、`ABO`、`3D-FUTURE`、`HSSD` 和 `Toys4k`。
- `OUTPUT_DIR`：保存数据的目录。
- `SOURCE`：如果 `SUBSET` 为 `ObjaverseXL` 则为必填项。选项 (options) 包括 `sketchfab` 和 `github`。

例如，要加载 ObjaverseXL (sketchfab) 子集 (subset) 的元数据 (metadata) 并将其保存到 `datasets/ObjaverseXL_sketchfab`，我们可以运行：

```
python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab
```

### 步骤 3：下载数据 (Download Data)

接下来，我们需要下载 3D 资产 (3D assets)。

```
python dataset_toolkits/download.py <SUBSET> --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`：要下载的数据集子集 (subset)。选项 (options) 包括 `ObjaverseXL`、`ABO`、`3D-FUTURE`、`HSSD` 和 `Toys4k`。
- `OUTPUT_DIR`：保存数据的目录。

如果您使用多个节点 (nodes) 进行数据准备，还可以指定当前进程 (process) 的 `RANK` (rank) 和 `WORLD_SIZE` (world_size)。

例如，要下载 ObjaverseXL (sketchfab) 子集 (subset) 并将其保存到 `datasets/ObjaverseXL_sketchfab`，我们可以运行：

***注意：下方示例命令出于演示目的设置了很大的 `WORLD_SIZE` (world_size)。将仅下载数据集 (dataset) 的一小部分。***

```
python dataset_toolkits/download.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --world_size 160000
```

某些数据集 (datasets) 可能需要交互式登录 (interactive login) 至 Hugging Face 或进行手动下载。请遵循工具包 (toolkits) 给出的指令。

下载完成后，使用以下命令更新元数据 (metadata) 文件：

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### 步骤 4：渲染多视角图像 (Render Multiview Images)（以及计算美学评分 (Calculate Aesthetic Scores)）

可以使用以下命令渲染多视角图像 (multiview images)：

```
python dataset_toolkits/render.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`：要渲染的数据集子集 (subset)。选项 (options) 包括 `ObjaverseXL`、`ABO`、`3D-FUTURE`、`HSSD` 和 `Toys4k`。
- `OUTPUT_DIR`：保存数据的目录。
- `NUM_VIEWS`：要渲染的视角数量。默认为 150。
- `RANK` (rank) 和 `WORLD_SIZE` (world_size)：多节点配置 (Multi-node configuration)。

例如，要渲染 ObjaverseXL (sketchfab) 子集 (subset) 并将其保存到 `datasets/ObjaverseXL_sketchfab`，我们可以运行：

```
python dataset_toolkits/render.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

（可选）如果您想计算自己渲染的数据集 (datasets) 的美学评分 (aesthetic scores)，可以使用以下命令：

```
python dataset_toolkits/calculate_aesthetic_scores.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```
- `OUTPUT_DIR`：保存数据的目录。
- `RANK` (rank) 和 `WORLD_SIZE` (world_size)：多节点配置 (Multi-node configuration)。

请不要忘记使用以下命令更新元数据 (metadata) 文件：

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### 步骤 5：体素化 3D 模型 (Voxelize 3D Models)

我们可以使用以下命令体素化 3D 模型 (3D models)：

```
python dataset_toolkits/voxelize.py <SUBSET> --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`：要体素化的数据集子集 (subset)。选项 (options) 包括 `ObjaverseXL`、`ABO`、`3D-FUTURE`、`HSSD` 和 `Toys4k`。
- `OUTPUT_DIR`：保存数据的目录。
- `RANK` (rank) 和 `WORLD_SIZE` (world_size)：多节点配置 (Multi-node configuration)。

例如，要体素化 ObjaverseXL (sketchfab) 子集 (subset) 并将其保存到 `datasets/ObjaverseXL_sketchfab`，我们可以运行：
```
python dataset_toolkits/voxelize.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

然后使用以下命令更新元数据 (metadata) 文件：

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### 步骤 6：提取 DINO 特征 (Extract DINO Features)

为了准备用于训练 SLat VAE 的训练数据，我们需要从多视角图像 (multiview images) 中提取 DINO 特征 (DINO features)，并将其聚合 (aggregate) 到稀疏体素网格 (sparse voxel grids) 中。

```
python dataset_toolkits/extract_features.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`：保存数据的目录。
- `RANK` (rank) 和 `WORLD_SIZE` (world_size)：多节点配置 (Multi-node configuration)。


例如，要从 ObjaverseXL (sketchfab) 子集 (subset) 中提取 DINO 特征 (DINO features) 并将其保存到 `datasets/ObjaverseXL_sketchfab`，我们可以运行：

```
python dataset_toolkits/extract_feature.py --output_dir datasets/ObjaverseXL_sketchfab
```

然后使用以下命令更新元数据 (metadata) 文件：

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### 步骤 7：编码稀疏结构 (Encode Sparse Structures)

将稀疏结构 (sparse structures) 编码为隐变量 (latents) 以训练第一阶段生成器 (first stage generator)：

```
python dataset_toolkits/encode_ss_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`：保存数据的目录。
- `RANK` (rank) 和 `WORLD_SIZE` (world_size)：多节点配置 (Multi-node configuration)。

例如，要将 ObjaverseXL (sketchfab) 子集 (subset) 的稀疏结构 (sparse structures) 编码为隐变量 (latents) 并将其保存到 `datasets/ObjaverseXL_sketchfab`，我们可以运行：

```
python dataset_toolkits/encode_ss_latent.py --output_dir datasets/ObjaverseXL_sketchfab
```

然后使用以下命令更新元数据 (metadata) 文件：

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### 步骤 8：编码 SLat (Encode SLat)

编码 SLat (Encode SLat) 以用于第二阶段生成器训练 (second stage generator training)：

```
python dataset_toolkits/encode_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`：保存数据的目录。
- `RANK` (rank) 和 `WORLD_SIZE` (world_size)：多节点配置 (Multi-node configuration)。

例如，要为 ObjaverseXL (sketchfab) 子集 (subset) 编码 SLat 并将其保存到 `datasets/ObjaverseXL_sketchfab`，我们可以运行：

```
python dataset_toolkits/encode_latent.py --output_dir datasets/ObjaverseXL_sketchfab
```

然后使用以下命令更新元数据 (metadata) 文件：

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### 步骤 9：渲染图像条件 (Render Image Conditions)

为了训练图像条件生成器 (image conditioned generator)，我们需要使用增强视角 (augmented views) 来渲染图像条件 (image conditions)。

```
python dataset_toolkits/render_cond.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`：要渲染的数据集子集 (subset)。选项 (options) 包括 `ObjaverseXL`、`ABO`、`3D-FUTURE`、`HSSD` 和 `Toys4k`。
- `OUTPUT_DIR`：保存数据的目录。
- `NUM_VIEWS`：要渲染的视角数量。默认为 24。
- `RANK` (rank) 和 `WORLD_SIZE` (world_size)：多节点配置 (Multi-node configuration)。

例如，要为 ObjaverseXL (sketchfab) 子集 (subset) 渲染图像条件 (image conditions) 并将其保存到 `datasets/ObjaverseXL_sketchfab`，我们可以运行：

```
python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

然后使用以下命令更新元数据 (metadata) 文件：

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```



