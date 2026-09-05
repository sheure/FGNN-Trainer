import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

import streamlit as st
import subprocess
import sys

# ---------- 设置页面 ----------
st.set_page_config(page_title="GNN 模型训练控制台", layout="wide")

# ---------- 启动锁 ----------
if "app_started" not in st.session_state:
    st.session_state.app_started = True
    print("=== app.py 首次启动 ===", flush=True)
    print(f"Python 版本: {sys.version}", flush=True)
    print(f"当前工作目录: {os.getcwd()}", flush=True)
    
    try:
        import torch
        print(f"torch 版本: {torch.__version__}", flush=True)
    except ImportError as e:
        print(f"torch 导入失败: {e}", flush=True)
    
    try:
        import numpy as np
        print(f"numpy 版本: {np.__version__}", flush=True)
    except ImportError as e:
        print(f"numpy 导入失败: {e}", flush=True)
else:
    print("=== app.py 已启动，跳过重复执行 ===", flush=True)

# ---------- 主界面 ----------
st.title("模型训练器（必须上传自定义数据）")

st.sidebar.header("数据与模型配置")

# ---------- 文件上传 ----------
uploaded_file = st.sidebar.file_uploader(
    "上传训练数据 (CSV)",
    type=["csv"],
    help="请上传 CSV 文件，必须包含 SMILES 和标签列。"
)

if uploaded_file is not None:
    st.sidebar.success(f"已上传文件：{uploaded_file.name}")

# ---------- 任务名称输入 ----------
task_name = st.sidebar.text_input("任务名称 (Task Name)", value="", help="例如：my_task")
st.sidebar.caption("任务名称必须手动输入，且不能为空")

# ---------- 训练超参数 ----------
with st.sidebar.expander("训练超参数"):
    epochs = st.number_input("训练轮数 (Epochs)", min_value=1, max_value=500, value=100, step=10)
    lr = st.number_input("学习率 (Learning Rate)", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    batch_size = st.number_input("批次大小 (Batch Size)", min_value=16, max_value=256, value=64, step=16)
    fp_dim = st.selectbox("指纹维度 (FP Dim)", options=[512, 700, 1024, 2513], index=3)
    split_type = st.selectbox("划分方式 (Split Type)", options=["random", "scaffold"], index=0)
    noise_rate = st.slider("噪声率 (Noise Rate)", 0.0, 0.5, 0.0, 0.05)

st.sidebar.markdown("---")

# ---------- 训练按钮 ----------
if st.sidebar.button("开始训练", type="primary"):
    if uploaded_file is None:
        st.error("请先上传 CSV 数据文件！")
        st.stop()

    if not task_name:
        st.error("请先输入任务名称！")
        st.stop()

    data_file = f"dataset/{task_name}/train.csv"
    try:
        os.makedirs(f"dataset/{task_name}", exist_ok=True)
        with open(data_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已将上传文件保存到 {data_file}")
    except Exception as e:
        st.error(f"保存上传文件失败：{e}")
        st.stop()

    graph_path = f"graph/{task_name}"
    if not os.path.exists(graph_path):
        os.makedirs(graph_path, exist_ok=True)
        st.info(f"创建图缓存目录：{graph_path}")

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
    st.info("正在启动训练，训练日志将实时显示在下方...")

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
        st.code(log_text, language="bash")出，错误码：{process.returncode}")
        st.code(log_text, language="bash")
