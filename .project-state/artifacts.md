# Artifacts

## ART-20260717-001 - FaceScape processed dataset

Description:
- 本地 FaceScape 规范化/预处理数据，是当前微调与审计主数据资源。
Path:
- `datasets/Facescape`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 包含 FaceScape train/test/merged 数据和预处理资源。
Notes:
- `du -sh` 显示约 441G；此前扫描确认 train/test 下有 `features`、`renders`、`renders_cond`、`voxels`。

## ART-20260717-002 - TRELLIS-image-large local model

Description:
- 本地 Hugging Face TRELLIS-image-large 模型目录和 checkpoint 权重。
Path:
- `microsoft/TRELLIS-image-large`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 训练、推理、预处理和重建工具引用的本地预训练模型资源。
Notes:
- `du -sh` 显示约 3.1G；存在 `ckpts/*.json` 和 `ckpts/*.safetensors`。

## ART-20260717-003 - 3D-FUTURE dataset resource

Description:
- 本地 TRELLIS 参考数据集 3D-FUTURE。
Path:
- `datasets/3D-FUTURE`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 9473 行；目录约 8.6M。

## ART-20260717-004 - ABO dataset resource

Description:
- 本地 TRELLIS 参考数据集 ABO。
Path:
- `datasets/ABO`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 4486 行；目录约 97M。

## ART-20260717-005 - HSSD dataset resource

Description:
- 本地 TRELLIS 参考数据集 HSSD。
Path:
- `datasets/HSSD`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 6671 行；目录约 6.0M。

## ART-20260717-006 - ObjaverseXL Sketchfab dataset resource

Description:
- 本地 TRELLIS 参考数据集 ObjaverseXL Sketchfab。
Path:
- `datasets/ObjaverseXL_sketchfab`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 168308 行；目录约 344M。

## ART-20260717-007 - Toys4k dataset resource

Description:
- 本地 TRELLIS 参考数据集 Toys4k。
Path:
- `datasets/Toys4k`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 3230 行；目录约 2.9M。
