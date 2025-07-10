from utils.file_utils import FileUtils

"""
if 'A' in f and result==0:
    predicted_count+=1
"""


def generate_sub_folder_map(target_sub_folder_file_path):
    sub_folder_list = FileUtils.read_file_as_lines(target_sub_folder_file_path)
    code_list = []
    for index, sub_folder_name in enumerate(sub_folder_list):
        if sub_folder_name.endswith("\n"):
            sub_folder_name = sub_folder_name.strip()
        # print(f"[{index}]{sub_folder_name}")
        if_part = f"if '{sub_folder_name}' in f and result=={index}:"
        if_statement = f"\tpredicted_count+=1"
        code_list.append(if_part)
        code_list.append(if_statement)
    code_content = "\n".join(code_list)
    return code_content


if __name__ == '__main__':
    target_sub_folder_file_path = "docs/sub_folder_list.txt"
    code_content = generate_sub_folder_map(target_sub_folder_file_path)
    print(code_content)
