"""
FaceScape 数据集处理的通用工具组件。

包含文件哈希计算、基于数字分段的拓扑寻址和物理路径直连定位函数。
"""

import hashlib
from pathlib import Path


def get_file_sha256(file_path: Path) -> str:
    """计算文件的 SHA-256 哈希值。

    Args:
        file_path (Path): 物理文件路径。

    Returns:
        str: 64 位 SHA-256 十六进制字符串。
    """
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_facescape_subfolder(subject_id: str) -> str:
    """依据确定的拓扑映射公式，将 subject_id (如 '1', '21') 定位到其所属的 FaceScape 分段目录。

    物理拓扑映射关系：
    - '1' ~ '20' -> '001-020'
    - '21' ~ '40' -> '021-040'
    - S_num -> (start)-(end)

    Args:
        subject_id (str): 样本 ID (如 '1', '21'...)。

    Returns:
        str: 分段文件夹名称 (如 '001-020', '021-040'...)，非数字返回空字符串。
    """
    if subject_id.isdigit():
        num = int(subject_id)
        start = ((num - 1) // 20) * 20 + 1
        end = start + 19
        return f"{start:03d}-{end:03d}"
    return ""


def get_subject_paths(dataset_root: Path, subject_id: str) -> tuple[Path, Path]:
    """根据确定性的物理映射，定位特定 Subject ID 的 Mesh 和相机参数目录。

    支持 FaceScape 标准分段目录结构以及扁平结构。

    Args:
        dataset_root (Path): 数据集根目录。
        subject_id (str): 样本 ID。

    Returns:
        tuple[Path, Path]: mesh_dir (Mesh目录) 和 camera_dir (相机参数目录)。
    """
    subfolder = get_facescape_subfolder(subject_id)
    if subfolder:
        mesh_dir = dataset_root / subfolder / "closed_shapes_meshlib" / subject_id
        camera_dir = dataset_root / subfolder / "aligned_camera_params" / subject_id
        if mesh_dir.exists():
            return mesh_dir, camera_dir

    # 扁平结构退避
    mesh_dir = dataset_root / "closed_shapes_meshlib" / subject_id
    camera_dir = dataset_root / "aligned_camera_params" / subject_id
    return mesh_dir, camera_dir
