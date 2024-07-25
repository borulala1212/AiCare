import streamlit as st

#Load data Eda Pkgs
import pandas as pd
import numpy as np

#load data viz pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

#Load data 
@st.cache

def load_data(HeartDisease):
	df = pd.read_csv(HeartDisease)
	return df

def run_eda_app_HD():
	st.subheader("Exploratory Data Analysis")
	df = load_data("HeartDisease/heart_disease_data.csv")
	freq_df = load_data("HeartDisease/freq_HD.csv")

	with st.expander("Data Source"):
		st.write("https://www.kaggle.com/code/harshalgadhe/heart-disease-uci")
	with st.expander("Heart Disease Dataset"):
		st.dataframe(df)


	with st.expander("Descriptive Summary"):
		st.dataframe(df.describe())

	with st.expander("Class Distribution"):
		st.dataframe(df['Class'].value_counts())

	with st.expander("Gender Distribution"):
		st.dataframe(df['Gender'].value_counts())

		#layouts
	col1,col2 = st.columns([2,1])

	with col1:
		with st.expander("Dist Plot of Gender"):
			fig = plt.figure()
			sns.countplot(df['Gender'])
			st.pyplot(fig)
			st.write("0 for Female; 1 for Male")
			gen_df = df['Gender'].value_counts().to_frame()

		with st.expander("Class Distribution"):
			st.dataframe(df['Class'].value_counts())




	with col2: 
		with st.expander("Dist Plot of Class"):
			fig = plt.figure()
			sns.countplot(df['Class'])
			st.pyplot(fig)
		with st.expander("Frequency Distribution of Age"):
			st.dataframe(freq_df)


	with st.expander("Correlation Plot"):
		corr_matrix = df.corr()
		fig = plt.figure(figsize=(20,10))
		sns.heatmap(corr_matrix,annot=True)
		st.pyplot(fig)































