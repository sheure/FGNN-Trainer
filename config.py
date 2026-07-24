import os

# 获取当前文件夹的路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型存放的文件夹
MODEL_DIR = os.path.join(BASE_DIR, "model_save")

# 等一下我们会在 model_save 里找模型，这里先占个位，待会我根据你的文件名帮你改
MODEL_FILE_NAME = "请稍后替换为实际模型文件名.pth"

# 数据存放路径（如果你需要读原始数据的话）
DATA_DIR = os.path.join(BASE_DIR, "Data")