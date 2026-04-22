import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import sys

# 强制后台绘图防止多线程崩溃
import matplotlib
matplotlib.use('Agg')

# ✨ 引入 TabICLv2 核心
import tabicl
from tabicl import TabICLClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 🌟 核心拦截补丁：伪造缺失的类和路径，彻底骗过 Pickle
# ==========================================
# 1. 路径重定向
missing_modules = [
    'tabicl.sklearn', 'tabicl.sklearn.classifier', 
    'tabicl.sklearn.preprocessing', 'tabicl.sklearn.metrics', 
    'tabicl.sklearn.utils', 'tabicl.ensemble'
]
for mod in missing_modules:
    if mod not in sys.modules:
        sys.modules[mod] = tabicl

# 2. 伪造 TransformToNumerical 类（它必须继承 sklearn 的基类）
class TransformToNumerical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# 3. 强行注入命名空间
tabicl.TransformToNumerical = TransformToNumerical

# ==========================================
# 0. 页面配置与高级 CSS 美化
# ==========================================
st.set_page_config(
    page_title="mGPS=2 Collapse Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Times New Roman', sans-serif;
    }
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 15px;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B4F72;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    [data-testid="stSidebar"] {
        background-color: #F8F9F9;
        border-right: 1px solid #E5E7E9;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        color: #C0392B; 
        font-weight: 900;
    }
    input[type="number"] {
        font-weight: bold;
        color: #154360;
        background-color: #F4F6F7;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 头部设计
# ==========================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
with col_title:
    st.title("Intelligent Warning Platform for Postoperative mGPS=2 Collapse in Colorectal Cancer")
    st.markdown("**(Powered by TabICLv2: A State-of-the-Art Tabular Foundation Model)**")

st.markdown("""
<div style='background-color: #EBF5FB; padding: 15px; border-radius: 10px; border-left: 5px solid #2980B9; margin-bottom: 25px;'>
    <span style='color: #154360; font-size: 15px;'>
    <b>📊 System Introduction:</b> Powered by <b>TabICLv2</b>, this platform integrates 8 refined preoperative clinical indicators to dynamically predict the risk of <b>severe postoperative systemic inflammatory and nutritional collapse (mGPS=2)</b> in patients with <b>Colorectal Cancer (CRC)</b>. It features real-time <b>SHAP (SHapley Additive exPlanations)</b> interpretations, providing surgeons with unprecedented accuracy and explainable decision support at the point of care.
    </span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心引擎加载
# ==========================================
@st.cache_resource 
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "tabicl_model.pkl") 
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    with open(model_path, 'rb') as f:
        # 在 pickle 加载时，它会自动去 tabicl 模块下找 TransformToNumerical
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"🚨 Model loading failed. Error details: {e}")
    st.stop()

# ==========================================
# 3. 侧边栏与主界面双向绑定 (🌟 严格匹配 Excel 表头)
# ==========================================
default_values = {
    'PA': 200.0, 'Age': 65.0, 'Fbg': 3.0, 
    'ALB': 38.0, 'ChE': 6000.0, 'Lymph_pct': 25.0, 
    'PLT': 200.0, 'Ca': 2.30
}

for key, val in default_values.items():
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = val
    if f"{key}_num" not in st.session_state:
        st.session_state[f"{key}_num"] = val

def sync_inputs(src_key, dest_key):
    st.session_state[dest_key] = st.session_state[src_key]

st.sidebar.markdown("### 🖥️ System Status")
st.sidebar.success("🟢 Core Engine: TabICLv2 Ready")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Rapid Parameter Adjustment")

with st.sidebar.expander("👤 Demographics & Nutrition", expanded=True):
    st.slider("Age (Years)", 18.0, 100.0, step=1.0, key="Age_slider", on_change=sync_inputs, args=("Age_slider", "Age_num"))
    st.slider("Prealbumin (PA) mg/L", 50.0, 500.0, step=1.0, key="PA_slider", on_change=sync_inputs, args=("PA_slider", "PA_num"))
    st.slider("Albumin (ALB) g/L", 10.0, 60.0, step=0.1, key="ALB_slider", on_change=sync_inputs, args=("ALB_slider", "ALB_num"))
    st.slider("Cholinesterase (ChE) U/L", 1000.0, 15000.0, step=100.0, key="ChE_slider", on_change=sync_inputs, args=("ChE_slider", "ChE_num"))

with st.sidebar.expander("🩸 Immuno-coagulation Profile", expanded=True):
    st.slider("Lymphocyte Percentage (Lymph%)", 5.0, 60.0, step=0.1, key="Lymph_pct_slider", on_change=sync_inputs, args=("Lymph_pct_slider", "Lymph_pct_num"))
    st.slider("Platelets (PLT) ×10^9/L", 50.0, 800.0, step=1.0, key="PLT_slider", on_change=sync_inputs, args=("PLT_slider", "PLT_num"))
    st.slider("Fibrinogen (Fbg) g/L", 1.0, 10.0, step=0.1, key="Fbg_slider", on_change=sync_inputs, args=("Fbg_slider", "Fbg_num"))
    st.slider("Serum Calcium (Ca) mmol/L", 1.50, 3.00, step=0.01, key="Ca_slider", on_change=sync_inputs, args=("Ca_slider", "Ca_num"))

st.markdown("### 👨‍⚕️ Clinical Parameter Input Matrix")
st.markdown("*(Enter exact values below, or use the sidebar sliders to adjust synchronously)*")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.number_input("Age (Years)", min_value=18.0, max_value=100.0, step=1.0, format="%.0f", key="Age_num", on_change=sync_inputs, args=("Age_num", "Age_slider"))
    st.number_input("PA (mg/L)", min_value=50.0, max_value=500.0, step=1.0, format="%.1f", key="PA_num", on_change=sync_inputs, args=("PA_num", "PA_slider"))
with col2:
    st.number_input("ALB (g/L)", min_value=10.0, max_value=60.0, step=0.1, format="%.1f", key="ALB_num", on_change=sync_inputs, args=("ALB_num", "ALB_slider"))
    st.number_input("ChE (U/L)", min_value=1000.0, max_value=15000.0, step=100.0, format="%.0f", key="ChE_num", on_change=sync_inputs, args=("ChE_num", "ChE_slider"))
with col3:
    st.number_input("Lymph (%)", min_value=5.0, max_value=60.0, step=0.1, format="%.1f", key="Lymph_pct_num", on_change=sync_inputs, args=("Lymph_pct_num", "Lymph_pct_slider"))
    st.number_input("PLT (×10^9/L)", min_value=50.0, max_value=800.0, step=1.0, format="%.0f", key="PLT_num", on_change=sync_inputs, args=("PLT_num", "PLT_slider"))
with col4:
    st.number_input("Fbg (g/L)", min_value=1.0, max_value=10.0, step=0.1, format="%.2f", key="Fbg_num", on_change=sync_inputs, args=("Fbg_num", "Fbg_slider"))
    st.number_input("Ca (mmol/L)", min_value=1.50, max_value=3.00, step=0.01, format="%.2f", key="Ca_num", on_change=sync_inputs, args=("Ca_num", "Ca_slider"))

expected_features = ['PA', 'Age', 'Fbg', 'ALB', 'ChE', 'Lymph%', 'PLT', 'Ca']

input_df = pd.DataFrame({
    'PA': [st.session_state["PA_num"]], 
    'Age': [st.session_state["Age_num"]], 
    'Fbg': [st.session_state["Fbg_num"]], 
    'ALB': [st.session_state["ALB_num"]],
    'ChE': [st.session_state["ChE_num"]], 
    'Lymph%': [st.session_state["Lymph_pct_num"]], 
    'PLT': [st.session_state["PLT_num"]], 
    'Ca': [st.session_state["Ca_num"]]
})

input_df = input_df[expected_features]

# ==========================================
# 4. 前向推理与 SHAP 动态解析 (TreeExplainer 稳健版)
# ==========================================
if st.button("🚀 Run Risk Assessment", type="primary"):
    with st.spinner('🧬 In-Context Learning model is analyzing clinical features...'):
        
        risk_prob = model.predict_proba(input_df)[0][1] 
        
        st.markdown("---")
        st.markdown("### 🎯 Postoperative Risk Inference Report")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Probability of mGPS=2 Collapse", value=f"{risk_prob * 100:.2f} %")
            
        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True) 
            if risk_prob > 0.5: 
                st.error("🚨 **[HIGH RISK ALERT]** The model identifies this patient as highly susceptible to **severe postoperative systemic inflammatory and nutritional collapse (mGPS=2)**. Intensive perioperative immunonutritional management and preemptive anti-inflammatory protocols are strongly recommended.")
                st.toast('High-risk alert detected!', icon='⚠️') 
            else:
                st.success("✅ **[SAFE ASSESSMENT]** The patient is currently in the low-risk zone. Maintenance of standard postoperative ERAS protocols is recommended.")
                st.balloons() 

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 Risk Factor Attribution (Real-time SHAP)")
        st.info("💡 **Interpretation Guide:** Explore different tabs to view the explanations. Red color indicates risk-increasing factors, while blue indicates protective factors.")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            if isinstance(shap_values, list):
                shap_val_single = shap_values[1][0] 
                base_val = explainer.expected_value[1]
            else:
                shap_val_single = shap_values[0]
                base_val = explainer.expected_value

            exp = shap.Explanation(values=shap_val_single, 
                                   base_values=base_val, 
                                   data=input_df.iloc[0], 
                                   feature_names=expected_features)
            
            tab1, tab2, tab3, tab4 = st.tabs(["🌊 Waterfall Plot", "⚖️ Force Plot", "📈 Decision Plot", "📊 Bar Plot"])
            
            with tab1:
                st.markdown("#### 1. Local Waterfall Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
                
            with tab2:
                st.markdown("#### 2. Local Force Plot")
                shap.force_plot(base_val, shap_val_single, input_df.iloc[0], matplotlib=True, show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()
                
            with tab3:
                st.markdown("#### 3. Decision Plot")
                shap.decision_plot(base_val, shap_val_single, input_df.iloc[0], show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()
                
            with tab4:
                st.markdown("#### 4. Absolute Impact Bar Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
            
        except Exception as e:
            st.error(f"An error occurred while generating the SHAP plots: {e}")
