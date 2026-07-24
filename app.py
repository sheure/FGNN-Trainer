import streamlit as st
import subprocess
import sys
import os
import threading
import time
import shutil

st.set_page_config(page_title="GNN 模型训练控制台", layout="wide")
st.title("模型训练器（基于你的 train.py）")

st.sidebar.header("数据与模型配置")

# 获取 dataset 目录下已有任务
dataset_path = "dataset"
if os.path.exists(dataset_path):
    available_tasks = [d for d in os.listdir(dataset_path) if
                       os.path.isdir(os.path.join(dataset_path, d)) and os.path.exists(
                           os.path.join(dataset_path, d, "train.csv"))]
else:
    available_tasks = []

task_name = st.sidebar.text_input("任务名称 (Task Name)", value="hiv" if "hiv" in available_tasks else "")
if available_tasks:
    st.sidebar.write("检测到已有数据集:", ", ".join(available_tasks))
st.sidebar.caption("要求：dataset/{任务名}/train.csv 必须存在")

# 训练参数（与你的 train.py 对应）
with st.sidebar.expander("训练超参数"):
    epochs = st.number_input("训练轮数 (Epochs)", min_value=1, max_value=500, value=100, step=10)
    lr = st.number_input("学习率 (Learning Rate)", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    batch_size = st.number_input("批次大小 (Batch Size)", min_value=16, max_value=256, value=64, step=16)
    fp_dim = st.selectbox("指纹维度 (FP Dim)", options=[512, 700, 1024, 2513], index=3)  # 默认2513
    split_type = st.selectbox("划分方式 (Split Type)", options=["random", "scaffold"], index=0)
    noise_rate = st.slider("噪声率 (Noise Rate)", 0.0, 0.5, 0.0, 0.05)

st.sidebar.markdown("---")
if st.sidebar.button("开始训练", type="primary"):
    if not task_name:
        st.error("请先输入任务名称！")
    else:
        data_file = f"dataset/{task_name}/train.csv"
        if not os.path.exists(data_file):
            st.error(f"找不到数据文件：{data_file}，请先放置训练数据！")
        else:
            st.success(f"数据文件已找到：{data_file}")

            # 自动创建 graph 目录（如果不存在）
            graph_path = f"graph/{task_name}"
            if not os.path.exists(graph_path):
                os.makedirs(graph_path, exist_ok=True)
                st.info(f"创建图缓存目录：{graph_path}")

            # 构造命令行参数（与你的 train.py 完全匹配）
            cmd = [
                sys.executable,
                "train.py",
                "--task_name", task_name,
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--batch_size", str(batch_size),
                "--fp_dim", str(fp_dim),
                "--split_type", split_type,
                "--noise_rate", str(noise_rate),
                "--graph_path", graph_path,
            ]

            st.code(" ".join(cmd), language="bash")
            st.info("正在启动训练，训练日志将实时显示在下方（GNN训练较慢，请耐心等待）...")

            log_placeholder = st.empty()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.getcwd()
            )

            log_text = ""
            with st.spinner("训练进行中..."):
                for line in iter(process.stdout.readline, ''):
                    if line:
                        log_text += line
                        # 只保留最近 100 行，防止页面卡顿
                        lines = log_text.split('\n')
                        if len(lines) > 100:
                            lines = lines[-100:]
                            log_text = '\n'.join(lines)
                        log_placeholder.code(log_text, language="bash")

                process.wait()

            if process.returncode == 0:
                st.success(f"训练完成！模型已保存到 model_save/{task_name}/")
            else:
                st.error(f"训练异常退出，错误码：{process.returncode}")
                st.code(log_text, language="bash")