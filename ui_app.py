import streamlit as st
import requests
import pandas as pd
import os
import json

st.set_page_config(page_title="HIV/GNN 预测系统", layout="wide")
st.title("分子属性预测系统")

st.sidebar.header("模型选择")

# 1. 扫描 model_save 下有哪些训练好的任务
model_base = "model_save"
if os.path.exists(model_base):
    tasks = [d for d in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, d))]
else:
    tasks = []

if tasks:
    selected_task = st.sidebar.selectbox("选择训练好的任务", tasks)

    # 扫描该任务下有哪些种子文件夹
    task_path = os.path.join(model_base, selected_task)
    seeds = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d)) and d.isdigit()]
    if seeds:
        selected_seed = st.sidebar.selectbox("选择随机种子 (Seed)", seeds)
    else:
        selected_seed = "90"
        st.sidebar.warning("未找到种子文件夹，默认使用 90")
else:
    st.sidebar.error("未找到 model_save 目录或没有训练好的模型，请先运行训练！")
    selected_task = "hiv"
    selected_seed = "90"

st.sidebar.markdown("---")
st.sidebar.header("上传预测数据")
st.sidebar.markdown("请上传 **CSV 文件**，格式需与训练数据一致（包含 SMILES 和特征列）")

uploaded_file = st.sidebar.file_uploader("点击选择 CSV 文件", type=["csv"])

# 主界面
st.subheader("预测结果")

# 显示当前使用的模型
st.info(f"当前模型：`{selected_task}` / Seed `{selected_seed}`")

if uploaded_file is not None:
    # 预览数据
    df = pd.read_csv(uploaded_file)
    st.write("已上传的数据预览：")
    st.dataframe(df.head())

    if st.button("开始预测", type="primary"):
        # 把文件发给后端
        files = {"file": uploaded_file.getvalue()}
        try:
            with st.spinner("模型推理中，请稍候..."):
                # 注意：这里我们并没有把 task 和 seed 传给后端，因为我们后端是启动时固定加载的。
                # 如果我们想动态切换，需要修改后端逻辑。但为了简单，我们统一保持后端加载最新的。
                response = requests.post("http://127.0.0.1:8000/predict", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("预测完成！")

                # 将结果转为 DataFrame 显示
                result_df = pd.DataFrame({
                    "SMILES": result["smiles"],
                    "预测值": result["predictions"],
                    "真实值": result["labels"] if result["labels"] else "无"
                })
                st.dataframe(result_df)

                # 提供下载
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("下载预测结果", data=csv, file_name="predict_results.csv", mime="text/csv")
            else:
                st.error(f"预测失败：{response.text}")
        except requests.exceptions.ConnectionError:
            st.error("无法连接到后端服务！请确保先启动了 api_server.py（在另一个窗口运行）")
else:
    st.info("请先在左侧侧边栏上传 CSV 文件")

st.sidebar.markdown("---")
st.sidebar.caption("提示：训练新模型请运行 'streamlit run train_ui.py'")