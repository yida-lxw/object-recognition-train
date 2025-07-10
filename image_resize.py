import os

from PIL import Image


def resize_images(root_dir, image_width: int = 224, image_height: int = 224):
    """
    将目录下的所有图片缩放至224x224像素

    参数:
        root_dir (str): 图片根目录路径
    """
    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # 遍历根目录下的所有子目录
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # 检查文件是否为图片
            if file.lower().endswith(valid_extensions):
                # 原始文件完整路径
                original_path = os.path.join(subdir, file)

                try:
                    # 打开图片并缩放
                    with Image.open(original_path) as img:
                        # 保持宽高比的缩放（可选）# 这种方式会保持比例
                        # img.thumbnail((image_width, image_height))

                        # 强制缩放至224x224（可能变形）
                        img_resized = img.resize((image_width, image_height), Image.Resampling.LANCZOS)

                        # 保存覆盖原文件（或可选择保存到新目录）
                        img_resized.save(original_path)
                        print(f"已缩放处理图片: {original_path}")

                except Exception as e:
                    print(f"处理失败 {original_path}: {str(e)}")


if __name__ == "__main__":
    image_parent_folder_path = "E:/dataset_0709/building/val"
    resize_images(image_parent_folder_path)
