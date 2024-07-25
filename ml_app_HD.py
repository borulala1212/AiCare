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
    - Age : Age in years
	- sex : Sex (1 = male; 0 = female)
	- cp  : Chest Pain Type
			Value 0 : Typical Angina
			Value 1 : Atypical Angina
			Value 2 : Non Anginal Pain
			Value 3 : Asymptomatic
	- trestbps : Resting blood pressure (in mm Hg on admission to the hospital)
	- chol : Serum cholestoral in mg/dl 
	- fbs  : (Fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
	- restecg : Resting electrocardiographic results
			Value 0 : Normal
			Value 1 : Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
			Value 2 : Showing probable or definite left ventricular hypertrophy by Estes' criteria
	- thalach : Maximum heart rate achieved
	- exang : Exercise induced angina (1 = yes; 0 = no)
	- oldpeak : ST depression induced by exercise relative to rest
	- slope : The slope of the peak exercise ST segment
			Value 0 : Upsloping
			Value 1 : Flat
			Value 2 : Downsloping
	- ca : Number of major vessels (0-3) colored by flourosopy
	- thal : 0 = Normal; 1 = Fixed Defect; 2 = Reversable Defect
	- target : 0 = No Heart Disease, 1 = Heart Disease


"""

gender_map = {"Female":0,"Male":1}
cp_map = {"Typical Angina":0,"Atypical Angina":1,"Non Anginal Pain":2,"Asymptomatic":3}
fbs_map = {"False":0,"True":1}
restecg_map = {"Normal":0,"Having ST-T Wave Abnormality":1,"Showing Probable  or Definite Left Ventricular Hypertrophy by Estes' Criteria":2}
exang_map = {"No":0,"Yes":1}
slope_map = {"Upsloping":0,"Flat":1,"Downsloping":2}
thal_map = {"Normal":1,"Fixed Defect":2,"Reversable Defect":3}
target_map = {"No Heart Disease":0,"Heart Disease":1}

def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value


## Load ML Models 
@st.cache(allow_output_mutation=True)
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def run_ml_app_HD():
	st.subheader("Machine Learning Heart Disease Prediction")
#	st.write("It is working")
#	st.success("it is so cool")

	with st.expander("Attribute Info"):
		st.markdown(attrib_info)


	# Layout
	col1, col2 = st.columns(2)

	with col1:
		age = st.number_input("Age",min_value=0.0,max_value=90.0,format="%f")
		sex = st.selectbox("Gender",("Female","Male"))
		cp = st.selectbox("Chest Pain Type",("Typical Angina","Atypical Angina","Non Anginal Pain","Asymptomatic"))
		trestbps = st.number_input("Resting Blood Pressure (mmHg)",min_value=0.0,max_value=200.0,format="%f") 
		chol = st.number_input("Cholestoral (mg/dl)",min_value=0.0,max_value=400.0,format="%f")
		fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",("False","True")) 
		restecg = st.selectbox("Resting Electrocardiographic Results",("Normal","Having ST-T Wave Abnormality","Showing Probable  or Definite Left Ventricular Hypertrophy by Estes' Criteria"))
		
	with col2: 
		thalach = st.number_input("Maximum Heart Rate Achieved",min_value=0.0,max_value=200.0,format="%f") 
		exang = st.selectbox("Exercise Induced Angina",("No","Yes"))
		oldpeak = st.number_input("ST Depression Induced",min_value=0.0,max_value=10.0,format="%f") 
		slope = st.selectbox("The Slope of The Peak Exercise ST Segment",("Upsloping","Flat","Downsloping"))
		ca = st.number_input("Number of Major Vessels",min_value=0.0,max_value=10.0,format="%f")
		thal = st.selectbox("Thalassemia",("Normal","Fixed Defect","Reversable Defect"))

	with st.expander("Selected Option"):
		result = {'Age':age,
		'Gender':sex,
		'Chest Pain Type':cp,
		'Resting Blood Pressure (mmHg)':trestbps,
		'Serum Cholestoral (mg/dl)':chol,
		'Fasting Blood Sugar > 120 mg/dl':fbs,
		'Resting Electrocardiographic Results':restecg,
		'Maximum Heart Rate Achieved':thalach,
		'Exercise Induced Angina':exang,
		'ST Depression Induced by Exercise Relative to Rest':oldpeak,
		'The Slope of The Peak Exercise ST Segment':slope,
		'Number of Major Vessels (0-3) Colored by Flourosopy':ca,
		'Thalassemia':thal}
		st.write(result)

		encoded_result = []
		for i in result.values():
			if type(i) == float:
				encoded_result.append(i)
			elif i in ["Female","Male"]:
				a = get_value(i,gender_map)
				encoded_result.append(a)
			elif i in ["Typical Angina","Atypical Angina","Non Anginal Pain","Asymptomatic"]:
				b = get_value(i,cp_map)
				encoded_result.append(b)
			elif i in ["False","True"]:
				c = get_value(i,fbs_map)
				encoded_result.append(c)
			elif i in ["Normal","Having ST-T Wave Abnormality","Showing Probable  or Definite Left Ventricular Hypertrophy by Estes' Criteria"]:
				d = get_value(i,restecg_map)
				encoded_result.append(d)
			elif i in ["Upsloping","Flat","Downsloping"]:
				e = get_value(i,slope_map)
				encoded_result.append(e)
			elif i in ["Normal","Fixed Defect","Reversable Defect"]:
				f = get_value(i,thal_map)
				encoded_result.append(f)
			else:
				encoded_result.append(get_fvalue(i))

	if st.button("Predict"):
#		st.write(encoded_result)
		with st.spinner('In Progress...'):
			time.sleep(1)

		with st.expander("Prediction Result"):
			single_sample = np.array(encoded_result).reshape(1,-1)
#		st.write(single_sample)

			model = load_model("HeartDisease/RandomForest_model_HeartDisease.pkl")
			prediction = model.predict(single_sample)
			pred_prob = model.predict_proba(single_sample)
#		st.write(prediction)
#		st.write(pred_prob)

			if prediction == 1:
				st.warning("HEART DISEASE POSITIVE".format(prediction[0]))
				tempHD_1 = """

			<h3 style="color:olive;text-align:center;">The test result show that you have potentially HEART DISEASE POSITIVE.</h3>
			<h4 style="color:olive;text-align:center;">It appear you need a help. </h4>
			<h4 style="color:olive;text-align:center;">Please contact the doctor for your safety immediately.</h4>
			</div>
			
				"""
				stc.html(tempHD_1)
			#pred_probanility_score = {"No Heart Disease":pred_prob[0][0]*100,
			#"Heart Disease":pred_prob[0][1]*100}
			#st.write(pred_probanility_score)

			else:
				st.success("HEART DISEASE NEGATIVE".format(prediction[0]))
				tempHD_2 = """

			<h3 style="color:green;text-align:center;">The test result show that you appear fine and HEART DISEASE NEGATIVE.</h3>
			<h4 style="color:green;text-align:center;">Please contact the doctor to observe safety precautions and be aware of the Heart Disease.</h4>
			</div>
				"""
				stc.html(tempHD_2)
			#pred_probanility_score = {"No Heart Disease":pred_prob[0][0]*100,
			#"Heart Disease":pred_prob[0][1]*100}
			#st.write(pred_probanility_score)













