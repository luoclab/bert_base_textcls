import os
import shutil

def copy_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for file_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)  # 保留元数据
            print(f"Copied: {src_file} -> {dest_file}")

# 示例用法
src_directory = "data/train/毕设论文200"
dest_directory = "data/train/train_V2.0_all/毕设论文"
copy_files(src_directory, dest_directory)
