# Importing dependencies
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Loading the Standard Scaler
# scaler = StandardScaler()
# Loading the model
loaded_model = pickle.load(open(r"C:\Users\USER\Documents\EMEKA'S CODING STUFF\BACKEND\PYTHON IN GENERAL\Data Science\Introduction To Machine Learning\Codes\Project 2\diabetes_trained.sav", "rb"))

# Creating a function for prediction
def diabetes_prediction(input_data):
    # Changing the input data into a numpy array
    input_numpy_arr = np.asarray(input_data)

    # Reshape the array as we are predicting one instance
    reshaped_input_numpy_arr = input_numpy_arr.reshape(1, -1)

    # Standardize the reshaped input data
    # std_data = scaler.fit_transform(reshaped_input_numpy_arr)

    prediction = loaded_model.predict(reshaped_input_numpy_arr)
    print(prediction)

    if prediction[0] == 0:
        return("Absence of diabetes")
    else:
        return("Presence of diabetes")


def main():
    # Giving a title for the app
    st.title("Diabetes Prediction Web App")
    
    # Getting input data from the user
    Pregnancies = st.text_input("Number of Pregnancies: ")
    Glucose = st.text_input("Glucose Level: ")
    BloodPressure = st.text_input("Blood Pressure value: ")
    SkinThickness = st.text_input("Skin Thickness value: ")
    Insulin = st.text_input("Insulin Level: ")
    BMI = st.text_input("BMI value: ")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value: ")
    Age = st.text_input("Age of the patient: ")
    
    # Code for prediction
    diagnosis = ""
    
    # Creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
    
if __name__ == "__main__":
    main()    
    