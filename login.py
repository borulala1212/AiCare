import streamlit as st
import pandas as pd

import streamlit.components.v1 as stc

# Import mini apps
from eda_app_diabetesNum import run_eda_app_diabetesNum
from ml_app_diabetesNum import run_ml_app_diabetesNum

from eda_app_liver import run_eda_app_liver
from ml_app_liver import run_ml_app_liver

from eda_app_BC import run_eda_app_BC
from ml_app_BC import run_ml_app_BC

from eda_app_HD import run_eda_app_HD
from ml_app_HD import run_ml_app_HD

from PIL import Image

html_temp = """
		<h2 style="color:black;text-align:justify;">AICare is a useful application for predicting chronic diseases based on the implementation of machine learning. </h2>
		"""
html_temp1 = """
			####<p1> AICare focus on chronic disease such as </p>
				- Breast Cancer Disease
				- Diabetes Disease
				- Heart Disease
				- Liver Disease


"""
desc_temp = """
			### Early Stage Disease Predictor App
			This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
			#### Datasource
				- https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App

			"""


# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data



def main():
	"""Simple Login App"""
	img = Image.open("logo.png")
	st.sidebar.image(img)
	#st.sidebar.title("Simple Login App")

	menu = ["AICare","Login","SignUp","Logout"]
	choice = st.sidebar.selectbox("Home",menu)


	if choice == "AICare":
		img = Image.open("COVERUTAMA.jpg")
		st.image(img,use_column_width=True)
	#	stc.html(html_temp)
		st.subheader("AICare")
		st.write("AICare is a useful application for predicting breast cancer, diabetes, heart disease, and liver disease based on the implementation of machine learning.")

	if choice == "Login":
		#st.sidebar.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				st.sidebar.success("Welcome {}!".format(username))
				menu99 = ["Disease Prediction","About Us"]
				option= st.sidebar.selectbox('Menu',menu99)

				if option == "Disease Prediction":
					menu1 = ["Breast Cancer Disease","Diabetes Disease","Heart Disease","Liver Disease"]
					choice1 = st.sidebar.selectbox('Our Facilities',menu1)

					## Breast Cancer
					desc_temp_BC = """
						Original Wisconsin Breast Cancer Database
						"""
					## Heart Disease
					desc_temp_HD = """
							Heart Disease Dataset
					"""
					## DIABETES DISEASE

					desc_temp_diabetes = """
						Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
						"""
					## penyakit hati 
					desc_temp_liver = """
						This data set contains 416 liver patient records and 167 non liver patient records.The data set was collected from north east of Andhra Pradesh, India. Selector is a class label used to divide into groups(liver patient or not). This data set contains 441 male patient records and 142 female patient records. 
						"""

					if choice1 == "Breast Cancer Disease":
						BC = ["Explore Data Analysis","Breast Cancer Prediction"]
						img3 = Image.open("BREASTCANCER.jpg")
						st.image(img3,use_column_width=True)
						st.markdown(desc_temp_BC,unsafe_allow_html=True)
						choice11 = st.radio('Features',BC)
						
				
						if choice11 == "Explore Data Analysis":
							run_eda_app_BC()
						elif choice11 == "Breast Cancer Prediction":
							run_ml_app_BC()
						else:
							pass

					elif choice1 == "Diabetes Disease":
						DD = ["Explore Data Analysis","Diabetes Prediction"]
						img1 = Image.open("DIABETES.jpg")
						st.image(img1,use_column_width=True)
						st.markdown(desc_temp_diabetes,unsafe_allow_html=True)						
						choice12 = st.radio('Features',DD)
						

						if choice12 == "Explore Data Analysis":	
							run_eda_app_diabetesNum()
						elif choice12 == "Diabetes Prediction":
							run_ml_app_diabetesNum()
						else:
							pass

					elif choice1 == "Heart Disease":
						HD = ["Explore Data Analysis","Heart Prediction"]
						img4 = Image.open("HEART.jpg")
						st.image(img4,use_column_width=True)
						st.markdown(desc_temp_HD,unsafe_allow_html=True)
						choice13 = st.radio('Features',HD)						
						
						if choice13 == "Explore Data Analysis":

							run_eda_app_HD()
						elif choice13 == "Heart Prediction":
							run_ml_app_HD()
						else:
							pass
					## LIVER DISEASE
					elif choice1 == "Liver Disease":
						DD = ["Explore Data Analysis","Liver Prediction"]
						img2 = Image.open("LIVER.jpg")
						st.image(img2,use_column_width=True)
						st.markdown(desc_temp_liver,unsafe_allow_html=True)
						choice14 = st.radio('Features',DD)
						

						if choice14 == "Explore Data Analysis":					
							run_eda_app_liver()
						elif choice14 == "Liver Prediction":
							run_ml_app_liver()
						else:
							pass

				else:
					st.subheader("AICare")
					st.write("Chronic diseases worsen the health of patients because they are not treated quickly and are expensive to treat. Therefore, it is very important to minimize the risk of chronic disease through early detection of chronic disease. The focus of AICare is chronic diseases such as breast cancer, diabetes, heart disease, and liver disease with implementation of machine learning")


				#task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
				#if task == "Add Post":
				#	st.subheader("Add Your Post")

				#elif task == "Analytics":
				#	st.subheader("Analytics")
				#elif task == "Profiles":
				#	st.subheader("User Profiles")
				#	user_result = view_all_users()
				#	clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
				#	st.dataframe(clean_db)
			else:
				st.warning("Incorrect Username/Password")





	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")

	elif choice == "Logout":
		img = Image.open("COVERUTAMA.jpg")
		st.image(img,use_column_width=True)
		st.write("Thenkyou")




if __name__ == '__main__':
	main()