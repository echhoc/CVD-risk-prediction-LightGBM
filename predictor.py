import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components
from lime.lime_tabular import LimeTabularExplainer  # 可保留以备后续拓展

# 设置页面标题
st.title("Prediction of Cardiovascular Risk in New–onset T2D")
st.caption("Based on TyG Index and Carotid Ultrasound Features")

# ===== 载入模型和测试数据 =====
model = joblib.load('LGB.pkl')                # 训练好的 LightGBM 模型
X_test = pd.read_csv('x_test.csv')            # 原始测试集数据用于获取默认值

# ===== 显示用特征名称（用户输入）=====
feature_names = [
    "Age (years)",
    "Hypertension",
    "IMT (mm)",
    "TyG index",
    "Carotid plaque burden",
    "Maximum plaque thickness (mm)"
]

# ===== 输入表单 =====
with st.form("input_form"):
    st.subheader("Please enter the following clinical and ultrasound features:")
    inputs = []

    for col in feature_names:
        if col == "Hypertension":
            inputs.append(st.selectbox(col, options=[0, 1], index=0))

        elif col == "Age (years)":
            min_val = int(X_test[col].min())
            max_val = 100
            default_val = int(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=1)
            )

        elif col == "Carotid plaque burden":
            min_val = int(X_test[col].min())
            max_val = 15
            default_val = int(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=1)
            )

        elif col == "Maximum plaque thickness (mm)":
            min_val = 0.0
            max_val = 7.0
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

        elif col == "IMT (mm)":
            min_val = 0.0
            max_val = 1.5
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

        elif col == "TyG index":
            min_val = 0.0
            max_val = 15.0
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.01, format="%.2f")
            )

    submitted = st.form_submit_button("Submit Prediction")

# ===== 预测与解释部分 =====
if submitted:
    input_data = pd.DataFrame([inputs], columns=feature_names)
    input_data = input_data.round(2)  # 显示保留两位小数
    st.subheader("Model Input Features")
    st.dataframe(input_data)

    # 确保模型输入的列顺序与训练一致
    model_input = input_data[feature_names]

    # ===== 模型预测 =====
    predicted_proba = model.predict_proba(model_input)[0]
    probability = predicted_proba[1] * 100

    # ==== 显示预测结果 ====
    st.subheader("Prediction Result & Explanation")
    st.markdown(f"**Estimated probability:** {probability:.1f}%")

    # ===== SHAP 可解释性分析 =====
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_input)

    if isinstance(shap_values, list):  # 二分类情况
        shap_value_sample = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_value_sample = shap_values
        expected_value = explainer.expected_value

    # ===== Force Plot 可视化（HTML方式）=====
    force_plot_html = shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value_sample,
        features=model_input
    )

    st.markdown("### SHAP Force Plot (Feature Impact)")
    components.html(force_plot_html.html(), height=300, scrolling=True)
