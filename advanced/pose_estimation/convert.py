import os
from ultralytics.data.converter import convert_coco

def convert_coco_to_yolo(json_dir, use_segments=False, use_keypoints=False):
    """
    使用 Ultralytics 内置函数将 COCO JSON 转换为 YOLO 格式。

    参数:
    json_dir (str): 包含 COCO JSON 文件的目录路径。
    use_segments (bool): 是否转换分割（segmentation）标注。默认为 False。
    use_keypoints (bool): 是否转换关键点（keypoints）标注。默认为 False。
    """

    convert_coco(labels_dir=json_dir, use_segments=use_segments, use_keypoints=use_keypoints)

    print(f"转换完成!YOLO 格式的标签已保存在 {os.path.join(os.path.dirname(json_dir), 'labels')} 目录中。")


if __name__ == '__main__':

    coco_json_directory = './Cats/annotations'

    # --- 选择要转换的标注类型 ---
    # 如果是目标检测，两者都设为 False
    # CONVERT_FOR_DETECTION = True
    # 如果是实例分割，将 use_segments 设为 True
    # CONVERT_FOR_SEGMENTATION = True
    # 如果是姿态估计，将 use_keypoints 设为 True
    CONVERT_FOR_POSE_ESTIMATION = True


    # --- 执行转换 ---
    print("开始将 COCO JSON 转换为 YOLO .txt 格式...")

    if CONVERT_FOR_POSE_ESTIMATION:
        print("模式: 姿态估计 (包含边界框和关键点)")
        convert_coco_to_yolo(json_dir=coco_json_directory, use_segments=False, use_keypoints=True)
   