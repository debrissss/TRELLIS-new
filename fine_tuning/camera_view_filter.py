import argparse
import json
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

# 将项目根目录加入 python 模块搜索路径，支持直接以脚本运行
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger

# 统一获取集中管理的日志实例，禁止使用 print()
logger = get_logger("camera_view_filter")


def compute_face_center(
    mode: str,
    ply_path: Path = None,
    camera_params: dict = None,
) -> np.ndarray:
    """根据指定模式计算人脸中心点坐标。

    Args:
        mode (str): 计算模式，可选值为 'origin', 'mesh_mean', 'mesh_bbox', 'camera_intersection'。
        ply_path (Path, optional): PLY 模型路径。
        camera_params (dict, optional): 相机参数字典。

    Returns:
        np.ndarray: 计算出的人脸中心 3D 坐标。
    """
    if mode == "origin":
        logger.debug("采用原点 [0.0, 0.0, 0.0] 作为人脸中心。")
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    elif mode == "mesh_mean":
        if not ply_path or not ply_path.exists():
            logger.warning(f"PLY 文件不存在: {ply_path}，人脸中心降级使用 [0.0, 0.0, 0.0]。")
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        try:
            import trimesh
            logger.debug(f"正在读取 3D 人脸模型以计算顶点均值中心: {ply_path}")
            mesh = trimesh.load(str(ply_path), process=False)
            center = mesh.vertices.mean(axis=0)
            logger.debug(f"计算得出顶点均值中心: {center}")
            return center
        except Exception as e:
            logger.error(f"读取 3D 模型计算顶点均值失败: {e}，降级使用 [0.0, 0.0, 0.0]。")
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    elif mode == "mesh_bbox":
        if not ply_path or not ply_path.exists():
            logger.warning(f"PLY 文件不存在: {ply_path}，人脸中心将降级使用 [0.0, 0.0, 0.0]。")
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        try:
            import trimesh
            logger.debug(f"正在读取 3D 人脸模型以计算包围盒中心: {ply_path}")
            mesh = trimesh.load(str(ply_path), process=False)
            center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2.0
            logger.debug(f"计算得出包围盒中心: {center}")
            return center
        except Exception as e:
            logger.error(f"读取 3D 模型计算包围盒中心失败: {e}，降级使用 [0.0, 0.0, 0.0]。")
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    elif mode == "camera_intersection":
        if not camera_params:
            logger.warning("相机参数为空，人脸中心降级使用 [0.0, 0.0, 0.0]。")
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        try:
            logger.debug("正在通过相机光轴最小二乘交汇点计算人脸/相机圆顶中心...")
            M = np.zeros((3, 3))
            b = np.zeros(3)
            I = np.identity(3)
            count = 0
            for cam_id, params in camera_params.items():
                if "Rt" not in params:
                    continue
                Rt = np.array(params["Rt"], dtype=np.float64)
                R = Rt[:, :3]
                t = Rt[:, 3]
                c_i = -np.dot(R.T, t)
                d_i = R[2, :]
                d_norm = np.linalg.norm(d_i)
                if d_norm > 1e-6:
                    d_i = d_i / d_norm
                A_i = I - np.outer(d_i, d_i)
                M += A_i
                b += np.dot(A_i, c_i)
                count += 1
            if count > 0:
                T = np.linalg.solve(M, b)
                logger.debug(f"计算得出相机光轴交汇中心: {T}")
                return T
            else:
                logger.warning("没有有效的相机参数用于计算交汇点，降级使用 [0.0, 0.0, 0.0]。")
                return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        except Exception as e:
            logger.error(f"通过相机光轴交汇点计算中心失败: {e}，降级使用 [0.0, 0.0, 0.0]。")
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    else:
        logger.warning(f"未知的中心计算模式: {mode}，默认使用 [0.0, 0.0, 0.0]。")
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)


def filter_camera_views(
    image_dir: Path,
    params_path: Path,
    ply_path: Path = None,
    center_mode: str = "camera_intersection",
    thresh_left_back: float = 100.0,
    thresh_right_back: float = 100.0,
    thresh_up: float = 40.0,
    thresh_down: float = 40.0,
    mask_color: list = [0, 0, 255],
    mask_alpha: float = 0.5,
) -> None:
    """计算各相机中心的偏航角(Yaw)与俯仰角(Pitch)，在四个指定方向（左后、右后、上、下）独立判定并应用掩膜。

    通过指定的人脸中心点模式，计算相机中心 C 相对于该中心点的方向向量，并将其分解为偏航角和俯仰角进行阈值过滤。

    Args:
        image_dir (Path): 原始多视角图片文件夹路径。
        params_path (Path): 相机参数的 json 文件路径。
        ply_path (Path, optional): PLY 模型路径。
        center_mode (str): 中心点计算模式。
        thresh_left_back (float): 左侧后方偏航角过滤阈值，单位为度，默认 100.0。
        thresh_right_back (float): 右侧后方偏航角过滤阈值，单位为度，默认 100.0。
        thresh_up (float): 上方俯仰角过滤阈值，单位为度，默认 40.0。
        thresh_down (float): 下方俯仰角过滤阈值，单位为度，默认 40.0。
        mask_color (list): 遮罩颜色，BGR 格式列表。默认为 [0, 0, 255]（红色）。
        mask_alpha (float): 遮罩透明度（不透明度），范围 [0, 1]。默认为 0.5。
    """
    # 1. 验证输入路径的合法性
    if not image_dir.exists() or not image_dir.is_dir():
        logger.error(f"输入的多视角图片文件夹不存在: {image_dir}")
        return

    if not params_path.exists() or not params_path.is_file():
        logger.error(f"输入的相机参数文件不存在: {params_path}")
        return

    # 2. 复制文件夹到同级目录下，并附加 _masked 后缀
    dst_dir = image_dir.parent / f"{image_dir.name}_masked"
    if dst_dir.exists():
        logger.warning(f"目标目录已存在: {dst_dir}，将直接在已有文件夹内更新遮罩图像。")
    else:
        try:
            shutil.copytree(str(image_dir), str(dst_dir))
            logger.info(f"成功复制文件夹: {image_dir} -> {dst_dir}")
        except Exception as e:
            logger.error(f"复制多视角图片文件夹失败: {e}")
            return

    # 3. 读取相机参数
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            raw_params = json.load(f)
    except Exception as e:
        logger.error(f"解析相机参数 json 失败: {e}")
        return

    # 4. 解析相机外参
    camera_params = {}
    for k, v in raw_params.items():
        if "_" in k:
            parts = k.split("_", 1)
            if parts[0].isdigit():
                cam_id = int(parts[0])
                field_name = parts[1]
                if cam_id not in camera_params:
                    camera_params[cam_id] = {}
                camera_params[cam_id][field_name] = v

    if not camera_params:
        logger.error("在 params.json 中未解析出任何有效的相机参数")
        return

    # 5. 获取计算出的人脸/正面中心点坐标
    face_center = compute_face_center(center_mode, ply_path, camera_params)

    logger.info(
        f"开始进行多方向视角过滤评估。阈值设置: 左后={thresh_left_back}°, 右后={thresh_right_back}°, 上={thresh_up}°, 下={thresh_down}°"
    )

    # 6. 遍历相机参数，计算每个视角的偏航角与俯仰角
    processed_count = 0
    masked_count = 0

    for cam_id, params in sorted(camera_params.items()):
        if "Rt" not in params:
            logger.warning(f"相机 #{cam_id} 缺少外参 Rt，跳过过滤")
            continue

        # 获取相机旋转矩阵 R 和平移向量 t
        Rt = np.array(params["Rt"], dtype=np.float64)
        R = Rt[:, :3]
        t = Rt[:, 3]

        # 求解相机中心在世界坐标系中的 3D 坐标 C = -R^T * t
        camera_center = -np.dot(R.T, t)

        # 获得相对于计算出的人脸中心点的相对向量
        relative_pos = camera_center - face_center
        x, y, z = relative_pos[0], relative_pos[1], relative_pos[2]

        # 计算偏航角 (Yaw): 投影在 XZ 平面上，相对于 Z 轴正轴的夹角
        yaw_deg = np.degrees(np.arctan2(x, z))

        # 计算俯仰角 (Pitch): 相对于 XZ 水平平面的仰角/俯角
        horizontal_dist = np.sqrt(x**2 + z**2)
        pitch_deg = np.degrees(np.arctan2(y, horizontal_dist))

        # 独立并行判定是否需要加入遮罩
        apply_mask = False
        reasons = []
        draw_texts = []

        # 判定 A：左侧后方 (Yaw <= -90°)
        if yaw_deg <= -90.0:
            if abs(yaw_deg) > thresh_left_back:
                apply_mask = True
                reasons.append(f"左后侧偏角 |Yaw|={abs(yaw_deg):.1f}° > {thresh_left_back}°")
                draw_texts.append(f"Left-Back Yaw: {abs(yaw_deg):.1f} deg (Thresh: {thresh_left_back:.1f} deg)")

        # 判定 B：右侧后方 (Yaw >= 90°)
        if yaw_deg >= 90.0:
            if yaw_deg > thresh_right_back:
                apply_mask = True
                reasons.append(f"右后侧偏角 |Yaw|={yaw_deg:.1f}° > {thresh_right_back}°")
                draw_texts.append(f"Right-Back Yaw: {yaw_deg:.1f} deg (Thresh: {thresh_right_back:.1f} deg)")

        # 判定 C：上方视角 (Pitch > 0°)
        if pitch_deg > 0.0:
            if pitch_deg > thresh_up:
                apply_mask = True
                reasons.append(f"上方仰角 Pitch={pitch_deg:.1f}° > {thresh_up}°")
                draw_texts.append(f"Up Pitch: {pitch_deg:.1f} deg (Thresh: {thresh_up:.1f} deg)")

        # 判定 D：下方视角 (Pitch < 0°)
        if pitch_deg < 0.0:
            if abs(pitch_deg) > thresh_down:
                apply_mask = True
                reasons.append(f"下方俯角 |Pitch|={abs(pitch_deg):.1f}° > {thresh_down}°")
                draw_texts.append(f"Down Pitch: -{abs(pitch_deg):.1f} deg (Thresh: -{thresh_down:.1f} deg)")

        # 寻找对应的图片文件
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            p = dst_dir / f"{cam_id}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            logger.warning(f"在目标目录中未找到相机 #{cam_id} 对应的图片文件")
            continue

        processed_count += 1

        # 6. 读取图像并绘制视角参数与遮罩说明（对所有图像均生效）
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"读取图片失败: {img_path}")
            continue

        # 仅对超限视角叠加半透明遮罩
        if apply_mask:
            overlay = img.copy()
            overlay[:] = mask_color
            cv2.addWeighted(overlay, mask_alpha, img, 1.0 - mask_alpha, 0, img)
            masked_count += 1
            logger.info(
                f"相机 #{cam_id:03d} -> [过滤遮罩] | Yaw={yaw_deg:6.1f}°, Pitch={pitch_deg:5.1f}° | 原因: {', '.join(reasons)}"
            )
        else:
            logger.info(
                f"相机 #{cam_id:03d} -> [保留原图] | Yaw={yaw_deg:6.1f}°, Pitch={pitch_deg:5.1f}°"
            )

        # 准备绘制在图像上的偏角与阈值信息（适用于所有图像）
        draw_texts = [
            f"Yaw: {yaw_deg:.1f} deg (L-Thresh: -{thresh_left_back:.1f}, R-Thresh: {thresh_right_back:.1f})",
            f"Pitch: {pitch_deg:.1f} deg (U-Thresh: {thresh_up:.1f}, D-Thresh: -{thresh_down:.1f})",
            f"Status: {'MASKED' if apply_mask else 'KEEP'}"
        ]

        # 在图像上依次绘制（白字/红绿字，带黑边阴影以确保可读性）
        y_offset = 120
        for i, text in enumerate(draw_texts):
            # 绘制黑色投影/描边层
            cv2.putText(
                img,
                text,
                (54, y_offset + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.4,
                (0, 0, 0),
                12,
                cv2.LINE_AA,
            )
            # 绘制前景字，Status 行根据结果显示红/绿，其余显示白色
            if i == 2:
                color = (0, 0, 255) if apply_mask else (0, 255, 0)  # BGR: 红/绿
            else:
                color = (255, 255, 255)  # BGR: 白
                
            cv2.putText(
                img,
                text,
                (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.4,
                color,
                6,
                cv2.LINE_AA,
            )
            y_offset += 100

        # 保存带文本和遮罩的最终图像
        cv2.imwrite(str(img_path), img)

    logger.info(
        f"多视角过滤处理完成！共检查 {processed_count} 张图片，其中 {masked_count} 张已叠加半透明遮罩。"
    )


def main() -> None:
    """主入口函数，解析多方向命令行阈值参数并启动视角过滤。"""
    parser = argparse.ArgumentParser(
        description="三维人脸多视角相机偏航角(Yaw)与俯仰角(Pitch)过滤及遮罩工具"
    )
    # 核心位置/名称参数
    parser.add_argument(
        "subject_dir",
        type=str,
        help="样本目录路径或名称（例如 301_1_neutral，或完整路径 /Users/lym/Downloads/301_1_neutral）",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="多视角图片文件夹路径（默认从 subject_dir 自动推导）",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default=None,
        help="相机标定参数的 JSON 文件路径（默认从 subject_dir 自动推导）",
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        default=None,
        help="三维人脸 PLY 模型文件路径（默认从 subject_dir 自动推导）",
    )
    parser.add_argument(
        "--center_mode",
        type=str,
        default="camera_intersection",
        choices=["origin", "mesh_mean", "mesh_bbox", "camera_intersection"],
        help="人脸中心计算模式：origin(使用原点[0,0,0])，mesh_mean(模型顶点均值)，mesh_bbox(模型包围盒中心)，camera_intersection(相机光轴交汇点中心，默认且推荐)",
    )
    parser.add_argument(
        "--thresh_left_back",
        type=float,
        default=100.0,
        help="左侧后方视角过滤的偏航角度阈值（默认 100.0）",
    )
    parser.add_argument(
        "--thresh_right_back",
        type=float,
        default=100.0,
        help="右侧后方视角过滤的偏航角度阈值（默认 100.0）",
    )
    parser.add_argument(
        "--thresh_up",
        type=float,
        default=40.0,
        help="上方视角过滤的俯仰角度阈值（默认 40.0）",
    )
    parser.add_argument(
        "--thresh_down",
        type=float,
        default=40.0,
        help="下方视角过滤的俯仰角度阈值（默认 40.0）",
    )
    parser.add_argument(
        "--mask_color",
        type=int,
        nargs=3,
        default=[0, 0, 255],
        help="遮罩颜色（BGR格式，三个整数，默认 [0, 0, 255] 红色）",
    )
    parser.add_argument(
        "--mask_alpha",
        type=float,
        default=0.5,
        help="遮罩的不透明度 Alpha（范围在 0.0 - 1.0 之间，默认 0.5）",
    )

    args = parser.parse_args()

    # 1. 尝试解析样本目录
    subject_dir = Path(args.subject_dir)
    if not subject_dir.exists():
        # 如果输入的不是合法直接路径，尝试拼接默认的 Downloads 路径
        default_base = Path("/Users/lym/Downloads")
        potential_dir = default_base / args.subject_dir
        if potential_dir.exists():
            subject_dir = potential_dir
        else:
            logger.error(
                f"输入的样本目录或名称不存在: {args.subject_dir}，且在默认下载目录 {default_base} 下也未找到。"
            )
            return

    # 2. 从目录名称中提取表情名（FaceScape 命名格式：{subject_id}_{expression_name}）
    if "_" in subject_dir.name:
        expr_name = subject_dir.name.split("_", 1)[1]
    else:
        expr_name = subject_dir.name

    # 3. 自动推导子物理要素的路径
    image_dir = Path(args.image_dir) if args.image_dir else (subject_dir / expr_name)
    params_path = Path(args.params_path) if args.params_path else (subject_dir / "params.json")
    ply_path = Path(args.ply_path) if args.ply_path else (subject_dir / f"{expr_name}.ply")

    logger.info(f"成功定位样本根目录: {subject_dir}")
    logger.info(f"-> 自动推导多视角图目录: {image_dir}")
    logger.info(f"-> 自动推导相机标定参数: {params_path}")
    logger.info(f"-> 自动推导 3D 网格模型: {ply_path}")

    filter_camera_views(
        image_dir=image_dir,
        params_path=params_path,
        ply_path=ply_path,
        center_mode=args.center_mode,
        thresh_left_back=args.thresh_left_back,
        thresh_right_back=args.thresh_right_back,
        thresh_up=args.thresh_up,
        thresh_down=args.thresh_down,
        mask_color=args.mask_color,
        mask_alpha=args.mask_alpha,
    )


if __name__ == "__main__":
    main()
