import streamlit as st
import subprocess
import sys
import os
import re
import time

# ==================== 路径配置（关键，先改这里） ====================
# app.py 所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 虚拟环境里的 python.exe（按你自己实际路径改）
PYTHON_EXE = r"D:\daima\FGNN_Project\FGNN-master\hiv_venv\Scripts\python.exe"
# train.py 的绝对路径
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train.py")


# ==================== 页面配置 ====================
st.set_page_config(page_title="GNN 模型训练控制台", layout="wide")
st.title("🧪 GNN 模型训练控制台")
st.caption("上传数据 · 配置超参数 · 实时监控训练过程")


# ==================== 全局状态（跨 session 共享） ====================
@st.cache_resource
def get_global_state():
    return {"is_training": False, "process": None, "cancel": False}

state = get_global_state()


# ==================== 侧边栏 ====================
st.sidebar.header("📂 数据与模型配置")

uploaded_file = st.sidebar.file_uploader(
    "上传训练数据 (CSV)",
    type=["csv"],
    help="请上传 CSV 文件，必须包含 SMILES 和标签列。"
)
if uploaded_file is not None:
    st.sidebar.success(f"已上传：{uploaded_file.name}")

task_name = st.sidebar.text_input(
    "任务名称 (Task Name)", value="",
    help="例如：my_task，将作为模型保存的文件夹名"
)
st.sidebar.caption("任务名称必须手动输入，且不能为空")

with st.sidebar.expander("⚙️ 训练超参数", expanded=True):
    epochs = st.number_input("训练轮数 (Epochs)", min_value=1, max_value=500, value=100, step=10)
    lr = st.number_input("学习率 (Learning Rate)", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    batch_size = st.number_input("批次大小 (Batch Size)", min_value=16, max_value=256, value=64, step=16)
    fp_dim = st.selectbox("指纹维度 (FP Dim)", options=[512, 700, 1024, 2513], index=3)
    split_type = st.selectbox("划分方式 (Split Type)", options=["random", "scaffold"], index=0)
    noise_rate = st.slider("噪声率 (Noise Rate)", 0.0, 0.5, 0.0, 0.05)

st.sidebar.markdown("---")

# 状态显示
if state["is_training"]:
    st.sidebar.error("🔥 训练进行中...")
else:
    st.sidebar.info("💤 空闲中，等待任务...")

# 按钮
col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("🚀 开始训练", type="primary", use_container_width=True)
cancel_btn = col2.button("🛑 取消训练", use_container_width=True)


# ==================== 主区：进度 + 日志 ====================
progress_bar = st.progress(0.0, text="训练进度：0%")
status_area = st.empty()
cmd_box = st.empty()
log_box = st.empty()


# ==================== 开始训练 ====================
if start_btn:
    if state["is_training"]:
        st.error("⚠️ 当前已有训练任务在运行，请先等待或取消。")
        st.stop()
    if uploaded_file is None:
        st.error("❌ 请先上传 CSV 数据文件！")
        st.stop()
    if not task_name:
        st.error("❌ 请先输入任务名称！")
        st.stop()

    # 1. 保存上传文件到 dataset/{task_name}/train.csv
    data_dir = os.path.join(BASE_DIR, "dataset", task_name)
    data_file = os.path.join(data_dir, "train.csv")
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(data_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ 已将上传文件保存到 {data_file}")
    except Exception as e:
        st.error(f"❌ 保存上传文件失败：{e}")
        st.stop()

    # 2. 创建 graph 目录
    graph_path = os.path.join(BASE_DIR, "graph", task_name)
    os.makedirs(graph_path, exist_ok=True)
    st.info(f"📁 创建图缓存目录：{graph_path}")

    # 3. 构造命令行（用虚拟环境 python + train.py 绝对路径）
    cmd = [
        PYTHON_EXE, TRAIN_SCRIPT,
        "--task_name", task_name,
        "--epochs", str(int(epochs)),
        "--lr", str(lr),
        "--batch_size", str(int(batch_size)),
        "--fp_dim", str(int(fp_dim)),
        "--split_type", split_type,
        "--noise_rate", str(noise_rate),
        "--graph_path", graph_path,
    ]
    cmd_str = " ".join(cmd)
    cmd_box.code(cmd_str, language="bash")
    st.info("⏳ 正在启动训练，训练日志将实时显示在下方（GNN 训练较慢，请耐心等待）...")

    # 4. 启动子进程
    state["is_training"] = True
    state["cancel"] = False
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=BASE_DIR,          # 关键：指定工作目录
        )
        state["process"] = process
    except Exception as e:
        state["is_training"] = False
        st.error(f"❌ 启动训练进程失败：{e}")
        st.stop()

    # 5. 实时读日志
    log_text = ""
    progress_pattern = re.compile(r'(\d+)/(\d+)')

    for line in iter(process.stdout.readline, ''):
        # 检查取消
        if state["cancel"]:
            log_text += "\n🛑 用户已取消训练，训练进程已被终止。\n"
            break

        if not line:
            break
        log_text += line

        # 只保留最后 200 行，避免页面卡顿
        lines = log_text.split('\n')
        if len(lines) > 200:
            lines = lines[-200:]
            log_text = '\n'.join(lines)

        log_box.code(log_text, language="bash")

        # 解析进度（形如 941/2036）
        m = progress_pattern.search(line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                pct = min(cur / total, 1.0)
                progress_bar.progress(pct, text=f"训练进度：{pct*100:.1f}%")

    process.wait()
    state["is_training"] = False
    state["process"] = None

    # 6. 训练结束处理
    if state["cancel"]:
        state["cancel"] = False
        st.warning("🛑 训练已取消。")
    elif process.returncode == 0:
        progress_bar.progress(1.0, text="训练进度：100%")
        st.success(f"✅ 训练完成！模型已保存到 model_save/{task_name}/")
    else:
        st.error(f"❌ 训练异常退出，错误码：{process.returncode}")


# ==================== 取消训练 ====================
if cancel_btn:
    if state["is_training"] and state["process"] is not None:
        state["cancel"] = True
        try:
            state["process"].terminate()
            time.sleep(0.5)
            if state["process"].poll() is None:
                state["process"].kill()
            st.warning("🛑 已强制终止训练进程。")
        except Exception as e:
            st.error(f"终止失败：{e}")
    else:
        st.info("ℹ️ 当前没有正在运行的任务。")
