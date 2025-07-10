import os

from utils.file_utils import FileUtils
from utils.string_utils import StringUtils


def generate_sub_folder_map(target_sub_folder_file_path):
    sub_folder_map = {}
    sub_folder_list = FileUtils.read_file_as_lines(target_sub_folder_file_path)
    for index, sub_folder_name in enumerate(sub_folder_list):
        sub_folder_map[index] = sub_folder_name
    return sub_folder_map

def generate_sub_folder_list(root_dir, target_sub_folder_file_path):
    sub_folder_list = []
    sub_folder_count = 0
    # 遍历根目录下的所有子目录
    for subdir, _, files in os.walk(root_dir):
        subdir = StringUtils.replaceBackSlash(subdir)
        sub_array = subdir.split("/")
        sub_array_len = len(sub_array)
        if sub_array_len <= 4:
            continue
        dot_index = subdir.rfind("/")
        sub_folder_name = subdir[dot_index + 1:]
        sub_folder_count = sub_folder_count + 1
        sub_folder_list.append(sub_folder_name)
        print(sub_folder_name)
    print(f"sub_folder count:{sub_folder_count}")

    sub_folder_content = "\n".join(sub_folder_list)
    FileUtils.write_string_to_file(sub_folder_content, target_sub_folder_file_path)


if __name__ == "__main__":
    target_sub_folder_file_path = "docs/sub_folder_list.txt"
    image_parent_folder_path = "E:/dataset_0709/building/train"
    generate_sub_folder_list(image_parent_folder_path, target_sub_folder_file_path)
