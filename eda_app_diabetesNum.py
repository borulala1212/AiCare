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
def load_data(DiabetesNumeric):
	df = pd.read_csv(DiabetesNumeric)
	return df

def run_eda_app_diabetesNum():
	st.subheader("Exploratory Data Analysis")
	df = load_data("DiabetesNumeric/Diabetes_clean.csv")
	freq_df = load_data("DiabetesNumeric/freq_AgeDB.csv")


	with st.expander("Data Source"):
		st.write("https://www.kaggle.com/datasets/mathchi/diabetes-data-set")
	with st.expander("Diabetes Disease Dataset"):
		st.dataframe(df)

	col1,col2 = st.columns([2,1])

	with col1:
		with st.expander("Distribution Plot of Class"):
			fig = plt.figure()
			sns.countplot(df['Class'])
			st.pyplot(fig)


	with col2: 
		with st.expander("Class Distribution"):
				st.dataframe(df['Class'].value_counts())

	with st.expander("Frequency Dist of Age"):
		p2 = px.bar(freq_df,x='Age',y='count')
		st.plotly_chart(p2)

	with st.expander("Descriptive Summary"):
		st.dataframe(df.describe())

	with st.expander("Correlation Plot"):
		corr_matrix = df.corr()
		fig = plt.figure(figsize=(20,10))
		sns.heatmap(corr_matrix,annot=True)
		st.pyplot(fig)































