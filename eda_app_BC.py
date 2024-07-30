import streamlit as st
import os

# Load data Eda Pkgs
import pandas as pd
import numpy as np

# Load data viz pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

# Load data with caching to avoid reloading data on every interaction
@st.cache
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    return pd.read_csv(file_path)

def run_eda_app_BC():
    st.subheader("Exploratory Data Analysis")
    try:
        df = load_data("BreastCancerWiscosin/Breast_Cancer_Wisconsin_clean.csv")
    except FileNotFoundError as e:
        st.error(f"Error: {str(e)}")
        st.stop()

    with st.expander("Data Source"):
        st.write("https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29")

    with st.expander("Breast Cancer Dataset"):
        st.dataframe(df)

    with st.expander("Descriptive Summary"):
        st.dataframe(df.describe())

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.expander("Dist Plot of Class"):
            fig = plt.figure()
            sns.countplot(df['Class'])
            st.pyplot(fig)
            st.write("0 for Benign; 1 for Malignant")

    with col2:
        with st.expander("Class Distribution"):
            st.dataframe(df['Class'].value_counts())

    # Correlation
    with st.expander("Correlation Plot"):
        corr_matrix = df.corr()
        fig = plt.figure(figsize=(20, 10))
        sns.heatmap(corr_matrix, annot=True)
        st.pyplot(fig)
