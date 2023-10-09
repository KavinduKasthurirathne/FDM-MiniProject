import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open('grid_search_gb_model.pkl', 'rb'))

with open('standard_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make predictions
def predict_loan_status(data):
    data['education'] = data['education'].map({'Not Graduate': 0, 'Graduate': 1})
    data['self_employed'] = data['self_employed'].map({'No': 0, 'Yes': 1})

    # Use the loaded scaler to transform new data
    new_data_scaled = scaler.transform(data)

    #predicting for the preprocessed data
    prediction = model.predict(new_data_scaled)

    return "Approved" if prediction[0] == 1 else "Rejected"

# Streamlit App
def main():
    st.title("Loan Status Prediction App")
    st.subheader("This app predicts the likelihood of loan approval based on applicant information and historical loan data.")
    st.write("Below are features used to predict the likelihood of loan approval")
    st.write("""<table>
            <tr><td> No of dependents</td><td> Number of dependents of the applicant</td></tr>
            <tr><td> Education</td><td>Education level of the applicant</td></tr>
            <tr><td> Self employed</td><td>If the applicant is self-employed or not</td></tr>
            <tr><td> Annual Income</td><td>Annual income of the applicant</td></tr>
            <tr><td> Loan Amount</td><td>Loan amount requested by the applicant</td></tr>
            <tr><td> Loan Term (Years)</td><td>Tenure of the loan requested by the applicant (in Years)</td></tr>
            <tr><td> CIBIL Score</td><td>CIBIL score of the applicant</td></tr>
            <tr><td> Residential Assets Value</td><td>Value of the residential asset of the applicant</td></tr>
            <tr><td> commercial_asset_value</td><td>Value of the commercial asset of the applicant</td></tr>
            <tr><td> luxury_asset_value</td><td>Value of the luxury asset of the applicant</td></tr>
            <tr><td> bank_assets_value</td><td>Value of the bank asset of the applicant</td></tr>
            </table><br>""", unsafe_allow_html=True)
    st.write("blue[Click the left side bar to insert information]")
    st.write("Please click the button below to see the loan status prediction")
    # Sidebar with user inputs

    st.sidebar.header("Insert the Applicant information")

    # Collect user input
    def user_input_features():
        no_of_dependents = st.sidebar.slider('Number of Dependents', 0, 10, 1)
        education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
        self_employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
        income_annum = st.sidebar.number_input('Annual Income', 0, step=1000)
        loan_amount = st.sidebar.number_input('Loan Amount', 0, step=1000)
        loan_term = st.sidebar.number_input('Loan Term (Years)', 2, 25, 2)
        cibil_score = st.sidebar.number_input('CIBIL Score', 300, 900, 600)
        residential_assets_value = st.sidebar.number_input('Residential Assets Value', 0, step=1000)
        commercial_assets_value = st.sidebar.number_input('Commercial Assets Value', 0, step=1000)
        luxury_assets_value = st.sidebar.number_input('Luxury Assets Value', 0, step=1000)
        bank_asset_value = st.sidebar.number_input('Bank Asset Value', 0, step=1000)

        feature_names = [
            'no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value'
        ]

        # Create a dictionary for user input
        data = {
            'no_of_dependents': no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value
        }

        # Check if all input values are provided
        if all(value is not None for value in data.values()):
            return pd.DataFrame(data, index=[0], columns=feature_names)
        else:
            return None

    # Get user input
    user_input = user_input_features()

    # Display user input
    # st.subheader('Your Input:')
    # st.write(user_input)

    # Button to trigger predictions
    if user_input is not None and st.button('Predict Loan Status'):
        # Make predictions and get labels
        prediction_label = predict_loan_status(user_input)

        # Display prediction label
        st.subheader('Prediction:')

        if prediction_label == "Approved":
            st.success("Loan will be Approved")
            st.write("Model Accuracy: 98.5%")
        else:
            st.error("Loan will be Rejected")
            st.write("Model Accuracy: 98.5%")

if __name__ == '__main__':
    main()
