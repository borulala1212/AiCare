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

def load_data(ILPD):
	df = pd.read_csv(ILPD)
	return df

def run_eda_app_liver():
	st.subheader("Exploratory Data Analysis")

	df = load_data("ILPD/Indian_Liver_Patient_Dataset_(ILPD).csv")

	df_encoded = load_data("ILPD/Indian_Liver_Patient_Disease_(ILPD)_clean.csv")
	freq_df = load_data("ILPD/freq_age.csv")

	with st.expander("Data Source"):
		st.write("https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)")
	with st.expander("Liver Disease Dataset"):
		st.dataframe(df)

	with st.expander("Descriptive Summary"):
		st.dataframe(df.describe())

		#layouts
	col1,col2 = st.columns([2,1])

	with col1:
		with st.expander("Dist Plot of Gender"):
			fig = plt.figure()
			sns.countplot(df['Gender'])
			st.pyplot(fig)

			gen_df = df['Gender'].value_counts().to_frame()

		with st.expander("Class"):
			fig = plt.figure()
			sns.countplot(df['Class'])
			st.pyplot(fig)


	with col2: 
		with st.expander("Gender Distribution"):
			st.dataframe(gen_df)

		with st.expander("Class Distribution"):
			st.dataframe(df['Class'].value_counts())

	with st.expander("Frequency Dist of Age"):
		p2 = px.bar(freq_df,x='age',y='count')
		st.plotly_chart(p2)

	with st.expander("Correlation Plot"):
		corr_matrix = df_encoded.corr()
		fig = plt.figure(figsize=(20,10))
		sns.heatmap(corr_matrix,annot=True)
		st.pyplot(fig)






























