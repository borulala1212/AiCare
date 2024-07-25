import streamlit as st
#Load ML Pkgs
import os
import joblib
# Load EDA Pkgs
import numpy as np
attrib_info = """
### Attribute Information:
    - Age 1.4-90
    - TB (Total Bilirubin)
    - DB (Direct Bilirubin)
    - Alkaline Phosphatase
    - Alamine Aminotransferase
    - Aspartate Aminotransferase
    - Total Proteins
    - Albumin
    - Ratio Albumin and Globulin Ratio
    - Selector Field 1. Liver Patient, 2. No Liver Patient
"""
#
#label_dict = {"No":0,"Yes":1}
gender_map = {"Female":0,"Male":1}
target_label_map = {"No Liver Patient":2,"Liver Patient":1}

#def get_fvalue(val):
#	feature_dict = {"No":0,"Yes":1}
#	for key,value in feature_dict.items():
#		if val == key:
#			return value

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value



## Load ML Models 
@st.cache(allow_output_mutation=True)
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def run_ml_app_liver():
	st.subheader("Machine Learning Liver Prediction")
#	st.write("It is working")
#	st.success("it is so cool")

	with st.expander("Attribute Info"):
		st.markdown(attrib_info)


	# Layout
	col1, col2 = st.columns(2)

	with col1:
		age = st.number_input("Age")
		#gender = st.selectbox("Gender",("Female","Male"))
		TB = st.number_input("Total Bilirubin")
		DB = st.number_input("Direct Bilirubin") 
		Alkhos = st.number_input("Alkaline Phosphatase")
		
	with col2:
		Sgpt = st.number_input("Alamine Aminotransferase") 
		Sgot = st.number_input("Aspartate Aminotransferase") 
		TP = st.number_input("Total Proteins") 
		ALB = st.number_input("Albumin")
		AG = st.number_input("Albumin and Globulin Ratio") 

	with st.expander("Your Selected Options"):
		result = {'Age':age,
		'Total Bilirubin':TB,
		'Direct Bilirubin':DB,
		'Alkaline Phosphatase':Alkhos,
		'Alamine Aminotransferase':Sgpt,
		'Aspartate Aminotransferase':Sgot,
		'Total Proteins':TP,
		'Albumin':ALB,
		'Albumin and Globulin Ratio':AG}
		st.write(result)
		#disini Gender dihapus

		a = [age,TB,DB,Alkhos,Sgpt,Sgot,TP,ALB,AG]

		encoded_result = []
		for i in result.values():
			#if type(i) == float:
			encoded_result.append(i)
			#elif i in ["Female","Male"]:
			#	res = get_value(i,gender_map)
			#	encoded_result.append(res)
			#else:
			#	encoded_result.append(get_fvalue(i))

		#st.write(encoded_result)


	with st.expander("Prediction result"):
		single_sample = np.array(encoded_result).reshape(1,-1)
#		st.write(single_sample)

		model = load_model("ILPD/RandomForest_model_ILPD.pkl")
		prediction = model.predict(single_sample)
		pred_prob = model.predict_proba(single_sample)
		#st.write(prediction)
		#st.write(pred_prob)

		if prediction == 1:
			st.warning("Liver Disease".format(prediction[0]))
			pred_probanility_score = {"No Liver Disease":pred_prob[0][0]*100,
			"Liver Disease":pred_prob[0][1]*100}
			st.write(pred_probanility_score)

		else:
			st.success("No Liver Disease".format(prediction[0]))
			pred_probanility_score = {"No Liver Disease":pred_prob[0][0]*100,
			"Liver Disease":pred_prob[0][1]*100}
			st.write(pred_probanility_score)


























