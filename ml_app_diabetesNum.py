import streamlit as st

#Load ML Pkgs
import os
import joblib
import time
import streamlit.components.v1 as stc
# Load EDA Pkgs
import numpy as np

attrib_info = """
#### Attribute Information:
	- Pregnancies	: Number of times pregnant 
	- Glucose		: Plasma glucose concentration a 2 hours in an oral glucose tolerance test 
	- BloodPressure	: Diastolic blood pressure (mm Hg) 
	- SkinThickness	: Triceps skin fold thickness (mm) 
	- Insulin		: 2-Hour serum insulin (mu U/ml) 
	- BMI			: Body mass index (weight in kg/(height in m)^2) 
	- DiabetesPedigreeFunction	: Diabetes pedigree function 
	- Age 			: Age (years) 
	- Outcome		: Class variable (0 or 1)

"""
# Load ML Models 
@st.cache(allow_output_mutation=True)
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def run_ml_app_diabetesNum():
	st.subheader("Machine Learning Diabetes Disease Prediction")
	#st.write("It is working")
	#st.success("it is so")

	with st.expander("Attribute Info"):
		st.markdown(attrib_info)


	# Layout
	col1, col2 = st.columns(2)

	with col1:
		Pregnancies = st.number_input("Pregnancies",0,17) 
		Glucose = st.number_input("Glucose",min_value=40.0,max_value=250.0,format="%f") 
		BloodPressure = st.number_input("BloodPressure",min_value=20.0,max_value=130.0,format="%f") 
		SkinThickness = st.number_input("SkinThickness",min_value=5.0,max_value=100.0,format="%f") 

	with col2:
		Insulin = st.number_input("Insulin",min_value=10.0,max_value=600.0,format="%f")
		BMI = st.number_input("BMI",min_value=10.0,max_value=70.0,format="%f") 
		DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction",min_value=0.0,max_value=5.0,format="%f") 
		Age = st.number_input("Age",10,100)  

	with st.expander("Selected Option"):
		result = {'Pregnancies':Pregnancies,
		'Glucose':Glucose,
		'BloodPressure':BloodPressure,
		'SkinThickness':SkinThickness,
		'Insulin':Insulin,
		'BMI':BMI,
		'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
		'Age':Age}

		st.write(result)

		encoded_result = []

		for i in result.values():
			#if type(i) == float:
			encoded_result.append(i)
	if st.button("Predict"):
		#st.write(encoded_result)
		with st.spinner('In Progress...'):
			time.sleep(1)

		with st.expander("Prediction result"):
			single_sample = np.array(encoded_result).reshape(1,-1)
		#st.write(single_sample)

			model = load_model("DiabetesNumeric/RandomForest_model_diabetes.pkl")
			prediction = model.predict(single_sample)
			pred_prob = model.predict_proba(single_sample)
		#st.write(prediction)
		#st.write(pred_prob)

			if prediction == 1:
				st.warning("DIABETES DISEASE POSITIVE".format(prediction[0]))
				tempD_1 = """

			<h3 style="color:olive;text-align:center;">The test result show that you have potentially DIABETES DISEASE POSITIVE.</h3>
			<h4 style="color:olive;text-align:center;">It appear you need a help. </h4>
			<h4 style="color:olive;text-align:center;">Please contact the doctor for your safety immediately.</h4>
			</div>
			
				"""
				stc.html(tempD_1)
			#pred_probanility_score = {"No Diabetes Disease":pred_prob[0][0]*100,
			#"Diabetes Disease":pred_prob[0][1]*100}
			#st.write(pred_probanility_score)

			else:
				st.success("DIABETES DISEASE NEGATIVE".format(prediction[0]))
				tempD_2 = """

			<h3 style="color:green;text-align:center;">The test result show that you appear fine and DIABETES DISEASE NEGATIVE.</h3>
			<h4 style="color:green;text-align:center;">Please contact the doctor to observe safety precautions and be aware of the Diabetes Disease.</h4>
			</div>
			
				"""
				stc.html(tempD_2)
			#pred_probanility_score = {"No Diabetes Disease":pred_prob[0][0]*100,
			#"Diabetes Disease":pred_prob[0][1]*100}
			#st.write(pred_probanility_score)













