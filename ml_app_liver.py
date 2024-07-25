import streamlit as st

#Load ML Pkgs
import os
import joblib
import time
import streamlit.components.v1 as stc

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
	st.subheader("Machine Learning Liver Disease Prediction")
	with st.expander("Attribute Info"):
		st.markdown(attrib_info)

	# Layout
	col1, col2 = st.columns(2)

	with col1:
		age = st.number_input("Age",0,100)
		#gender = st.selectbox("Gender",("Female","Male"))
		TB = st.number_input("Total Bilirubin",min_value=0.0,max_value=40.0,format="%f")
		DB = st.number_input("Direct Bilirubin",min_value=0.0,max_value=20.0,format="%f") 
		Alkhos = st.number_input("Alkaline Phosphatase",min_value=0.0,max_value=2200.0,format="%f")
		
	with col2:
		Sgpt = st.number_input("Alamine Aminotransferase",min_value=10.0,max_value=1500.0,format="%f") 
		Sgot = st.number_input("Aspartate Aminotransferase",min_value=0.0,max_value=1700.0,format="%f") 
		TP = st.number_input("Total Proteins",min_value=0.0,max_value=10.0,format="%f") 
		ALB = st.number_input("Albumin",min_value=0.0,max_value=10.0,format="%f")
		AG = st.number_input("Albumin and Globulin Ratio",min_value=0.0,max_value=10.0,format="%f") 

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
	if st.button("Predict"):
		#st.write(encoded_result)
		with st.spinner('In Progress...'):
			time.sleep(1)

		with st.expander("Prediction result"):
			single_sample = np.array(encoded_result).reshape(1,-1)
#		st.write(single_sample)

			model = load_model("ILPD/RandomForest_model_ILPD.pkl")
			prediction = model.predict(single_sample)
			pred_prob = model.predict_proba(single_sample)

			if prediction == 1:
				st.warning("LIVER DISEASE POSITIVE".format(prediction[0]))
				tempL_1 = """

			<h3 style="color:olive;text-align:center;">The test result show that you have potentially LIVER DISEASE POSITIVE.</h3>
			<h4 style="color:olive;text-align:center;">It appear you need a help. </h4>
			<h4 style="color:olive;text-align:center;">Please contact the doctor for your safety immediately.</h4>
			</div>
			
				"""
				stc.html(tempL_1)

			else:
				st.success("LIVER DISEASE NEGATIVE".format(prediction[0]))
				tempL_2 = """

			<h3 style="color:green;text-align:center;">The test result show that you appear fine and LIVER DISEASE NEGATIVE.</h3>
			<h4 style="color:green;text-align:center;">Please contact the doctor to observe safety precautions and be aware of the Liver Disease.</h4>
			</div>
			
				"""
				stc.html(tempL_2)













