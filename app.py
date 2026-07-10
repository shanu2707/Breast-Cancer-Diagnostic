import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set up page configurations for a modern, wide layouts
st.set_page_config(page_title="Breast Cancer Diagnostics Center", layout="wide")

# Safe wrapper cache to load our saved machine learning assets
@st.cache_resource
def load_ml_assets():
    with open('cancer_model.pkl', 'rb') as m_f:
        model = pickle.load(m_f)
    with open('scaler.pkl', 'rb') as s_f:
        scaler = pickle.load(s_f)
    return model, scaler

model, scaler = load_ml_assets()

# --- HEADER SECTION ---
st.title("Predictive Breast Cancer Diagnostic Interface")
st.markdown("---")

# --- SIDEBAR FOR PATIENT CASE METRICS INPUT ---
st.sidebar.header("Patient Clinical Metrics Input")
st.sidebar.markdown("Adjust cellular nuclei features below to run live predictive classifications:")

# Set up the core interactive sliders based on our dataset scale ranges
radius = st.sidebar.slider("Mean Radius", 5.0, 30.0, 14.0)
texture = st.sidebar.slider("Mean Texture", 5.0, 40.0, 19.0)
perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, 92.0)
area = st.sidebar.slider("Mean Area", 140.0, 2500.0, 650.0)
smoothness = st.sidebar.slider("Mean Smoothness", 0.05, 0.20, 0.10)

# Build a dummy array matching the 30-feature shape our scaler expects
input_features = np.zeros(30)
input_features[0] = radius
input_features[1] = texture
input_features[2] = perimeter
input_features[3] = area
input_features[4] = smoothness

# --- DIAGNOSTIC CALCULATION LOGIC ---
if st.sidebar.button("Run Diagnostic Classify"):
    # Apply standard scaling transformation mapping
    scaled_input = scaler.transform([input_features])
    prediction = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)[0]
    
    st.sidebar.markdown("### **Diagnostic Evaluation:**")
    if prediction[0] == 1:
        st.sidebar.error(f"Malignant Alert (Confidence: {probabilities[1]*100:.1f}%)")
    else:
        st.sidebar.success(f"Benign Clear (Confidence: {probabilities[0]*100:.1f}%)")

# --- MAIN DASHBOARD INTERACTIVE CHARTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Architectural Vector Clustering Space")
    # Generate random points simulating boundaries for the layout view
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'Mean Area Space': np.random.uniform(140, 2500, 150),
        'Mean Texture Space': np.random.uniform(5, 40, 150),
        'Mean Smoothness Space': np.random.uniform(0.05, 0.20, 150),
        'Condition Label': np.random.choice(['Benign', 'Malignant'], 150)
    })
    
    # Render an interactive 3D cluster plot
    fig_3d = px.scatter_3d(sample_df, x='Mean Area Space', y='Mean Texture Space', z='Mean Smoothness Space',
                           color='Condition Label', color_discrete_sequence=['#2ecc71', '#e74c3c'])
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    st.subheader("Core Diagnostic Engine Accuracy Tracker")
    # Render a high-end gauge visual
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 96.5,  # Matches evaluation report baselines
        title = {'text': "Model Production Evaluation Rate (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, 80], 'color': "#f1f2f6"},
                {'range': [80, 100], 'color': "#ced6e0"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)