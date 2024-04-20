# importing the libraries
# import pandas as pd
import pickle
import streamlit as st
import random as ran

# loading the models

breast_cancer = pickle.load(open("breast_cancer_model.sav", "rb"))

# sidebar for navigation
# Breast Cancer Prediction Page:


# page title
st.title("Breast Cancer Prediction using Machine Learning")

# getting the input data from the user

col1, col2, col3, col4, col5 = st.columns(5)

# with col1:
#     _id = st.number_input("id")

with col2:
    radius_mean = st.number_input("radius_mean")

with col3:
    texture_mean = st.number_input("texture_mean")

with col4:
    perimeter_mean = st.number_input("perimeter_mean")

with col5:
    area_mean = st.number_input("area_mean")

with col1:
    smoothness_mean = st.number_input("smoothness_mean")

with col2:
    compactness_mean = st.number_input("compactness_mean")

with col3:
    concavity_mean = st.number_input("concavity_mean")

with col4:
    concave_points_mean = st.number_input("concave points_mean")

with col5:
    symmetry_mean = st.number_input("symmetry_mean")

with col1:
    fractal_dimension_mean = st.number_input("fractal_dimension_mean")

with col2:
    radius_se = st.number_input("radius_se")

with col3:
    texture_se = st.number_input("texture_se")

with col4:
    perimeter_se = st.number_input("perimeter_se")

with col5:
    area_se = st.number_input("area_se")

with col1:
    smoothness_se = st.number_input("smoothness_se")

with col2:
    compactness_se = st.number_input("compactness_se")

with col3:
    concavity_se = st.number_input("concavity_se")

with col4:
    concave_points_se = st.number_input("concave points_se")

with col5:
    symmetry_se = st.number_input("ssymmetry_se")

with col1:
    fractal_dimension_se = st.number_input("fractal_dimension_se")

with col2:
    radius_worst = st.number_input("radius_worst")

with col3:
    texture_worst = st.number_input("texture_worst")

with col4:
    perimeter_worst = st.number_input("perimeter_worst")

with col5:
    area_worst = st.number_input("area_worst")

with col1:
    smoothness_worst = st.number_input("smoothness_worst")

with col2:
    compactness_worst = st.number_input("compactness_worst")

with col3:
    concavity_worst = st.number_input("concavity_worst")

with col4:
    concave_points_worst = st.number_input("concave points_worst")

with col5:
    symmetry_worst = st.number_input("symmetry_worst")

with col1:
    fractal_dimension_worst = st.number_input("fractal_dimension_worst")

# code for Prediction
breast_cancer_check = " "
_id = ran.randint(1000, 70000)
if st.button("Breast Cancer Test Result"):
    breast_cancer_prediction = breast_cancer.predict(
        [
            [
                _id,
                radius_mean,
                texture_mean,
                perimeter_mean,
                area_mean,
                smoothness_mean,
                compactness_mean,
                concavity_mean,
                concave_points_mean,
                symmetry_mean,
                fractal_dimension_mean,
                radius_se,
                texture_se,
                perimeter_se,
                area_se,
                smoothness_se,
                compactness_se,
                concavity_se,
                concave_points_se,
                symmetry_se,
                fractal_dimension_se,
                radius_worst,
                texture_worst,
                perimeter_worst,
                area_worst,
                smoothness_worst,
                compactness_worst,
                concavity_worst,
                concave_points_worst,
                symmetry_worst,
                fractal_dimension_worst,
            ]
        ]
    )
    print(breast_cancer_prediction)
    if breast_cancer_prediction[0] == 0:

        breast_cancer_check = "Patient don't have Breast Cancer."
        st.success(breast_cancer_check)
    else:
        breast_cancer_check = "Patient have Breast Cancer."
        st.error(breast_cancer_check)
