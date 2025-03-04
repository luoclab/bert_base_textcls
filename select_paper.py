import os
import shutil

def move_matching_txt_files(source_folder, target_folder, keywords, char_limit=200):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)  # 如果目标文件夹不存在，则创建
    
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):  # 仅处理txt文件
            file_path = os.path.join(source_folder, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read(char_limit)  # 读取前200个字符
                content = content.replace(" ", "")  # 去掉空格
                
                # 检查是否包含关键字
                if any(keyword in content for keyword in keywords):
                    shutil.move(file_path, os.path.join(target_folder, filename))  # 移动文件
                    print(f"Moved: {filename}")
    
source_folder = r"data/train/train_V2.0_all/论文文献"  # 你的源文件夹
target_folder = r"data/train/毕设论文1"  # 你的目标文件夹
os.makedirs(target_folder,exist_ok=True)

keywords = ["硕士学位", "毕业设计","毕业论文","学位论文"]

move_matching_txt_files(source_folder, target_folder, keywords)