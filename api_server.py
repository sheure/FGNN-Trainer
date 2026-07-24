import sys
import os
import shutil
import tempfile
import copy
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ---------- 重要：把当前目录加入系统路径，让 Python 能找到你的项目模块 ----------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------- 导入你项目里的核心工具 ----------
from tool import load_data, load_model, load_args, feature_selected, compute_score
from Data.Dataset import MixDataset
from torch.utils.data import DataLoader

# ---------- 初始化 ----------
app = FastAPI(title="HIV预测软件API")

# 全局变量
model = None
model_args = None


# ---------- 自动寻找最新训练好的模型 ----------
def find_latest_model():
    model_base = "model_save"
    if not os.path.exists(model_base):
        return None, None

    # 获取所有任务文件夹
    tasks = [d for d in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, d))]
    if not tasks:
        return None, None

    # 按修改时间排序，取最新的任务
    tasks.sort(key=lambda x: os.path.getmtime(os.path.join(model_base, x)), reverse=True)
    latest_task = tasks[0]

    task_path = os.path.join(model_base, latest_task)
    seeds = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d)) and d.isdigit()]
    if not seeds:
        return latest_task, None

    # 同样取最新的种子
    seeds.sort(key=lambda x: os.path.getmtime(os.path.join(task_path, x)), reverse=True)
    latest_seed = seeds[0]

    return latest_task, latest_seed


TASK_NAME, SEED = find_latest_model()
if TASK_NAME is None or SEED is None:
    print("警告：未找到任何训练好的模型，请先运行 train_ui.py 训练！")
    TASK_NAME = "hiv"
    SEED = "90"
else:
    print(f"自动加载最新模型：任务={TASK_NAME}，种子={SEED}")
BATCH_SIZE = 64  # ← 可根据情况调整


# ---------- 启动时自动加载模型 ----------
@app.on_event("startup")
def load_startup_model():
    global model, model_args
    # 模型路径：model_save/hiv/90/model.pt
    model_path = f"model_save/{TASK_NAME}/{SEED}/model.pt"

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请检查 model_save 文件夹结构是否与预期一致")
        return

    print("正在加载模型，请稍候...")
    # 1. 加载训练时的配置参数
    model_args = load_args(model_path)
    # 2. 加载模型权重
    model = load_model(model_args)
    model.eval()
    print(f"模型加载成功！路径：{model_path}")
    print(f"   任务：{TASK_NAME}，种子：{SEED}")


# ---------- 预测接口 ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    接收一个 CSV 文件，返回预测结果
    要求：CSV 格式与你训练时的 dataset/hiv/train.csv 一致
    """
    global model, model_args
    if model is None:
        raise HTTPException(status_code=500, detail="模型尚未加载，请检查 model_save 目录")

    # 1. 保存用户上传的文件到临时位置
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # 2. 用你项目里的 load_data 读取数据
        # 注意：load_data 需要 args 对象，我们复制一份并修改数据路径
        tmp_args = copy.deepcopy(model_args)
        tmp_args.predict_path = tmp_path  # 你的 predict.py 里用的是 predict_path 这个属性

        # 3. 加载数据并进行特征选择（和你 predict.py 里的流程一致）
        test_data = load_data(tmp_args, tmp_path)  # 注意：你的 load_data 可能需要两个参数
        test_data = feature_selected(test_data, tmp_args)

        # 4. 打包成 DataLoader
        test_dataset = MixDataset(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # 5. 执行预测（调用 predict.py 里的 predict 函数）
        from predict import predict as my_predict  # 导入你自己的预测函数
        test_pred = my_predict(model, test_dataloader, tmp_args)
        test_pred = np.array(test_pred).tolist()

        # 6. 提取 SMILES 和真实标签（如果有）
        smiles = [x.smile for x in test_data]
        targets = [x.label.tolist() if hasattr(x.label, 'tolist') else x.label for x in test_data]

        # 7. 返回结果
        return JSONResponse(content={
            "status": "success",
            "smiles": smiles,
            "predictions": test_pred,
            "labels": targets,
            "num_samples": len(test_data)
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测失败：{str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------- 健康检查 ----------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


# ---------- 启动服务器 ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)