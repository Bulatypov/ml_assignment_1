import streamlit as st
import requests

st.title("Boston Housing Price Prediction")

features = []
features.append(st.number_input("CRIM: Per capita crime rate by town", value=0.0))
features.append(st.number_input("ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.", value=0.0))
features.append(st.number_input("INDUS: Proportion of non-retail business acres per town", value=0.0))
features.append(st.number_input("CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)", value=0))
features.append(st.number_input("NOX: Nitric oxides concentration (parts per 10 million)", value=0.0))
features.append(st.number_input("RM: Average number of rooms per dwelling", value=0.0))
features.append(st.number_input("AGE: Proportion of owner-occupied units built prior to 1940", value=0.0))
features.append(st.number_input("DIS: Weighted distances to five Boston employment centers", value=0.0))
features.append(st.number_input("RAD: Index of accessibility to radial highways", value=0))
features.append(st.number_input("TAX: Full-value property tax rate per $10,000", value=0.0))
features.append(st.number_input("PTRATIO: Pupil-teacher ratio by town", value=0.0))
features.append(st.number_input("B: \(1000(Bk - 0.63)^2\) where Bk is the proportion of Black residents by town", value=0.0))
features.append(st.number_input("LSTAT: Percentage of lower status of the population", value=0.0))


if st.button("Predict"):
    response = requests.post("http://api:8000/predict/", json={"features": features})
    prediction = response.json().get("prediction")
    st.write(f"Predicted price: ${prediction:.2f}")
