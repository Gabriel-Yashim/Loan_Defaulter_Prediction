import streamlit as st
import numpy as np
import xgboost
import pickle

# Load the trained XGBoost model (assuming it's saved as 'xgboost_model.json')
model = pickle.load(open('XGBoost_model.pkl', 'rb'))

def preprocess_data(input_data):
    grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    subgrade_mapping = {
        'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4, 'B1': 5, 'B2': 6, 'B3': 7, 'B4': 8, 'B5': 9,
        'C1': 10, 'C2': 11, 'C3': 12, 'C4': 13, 'C5': 14, 'D1': 15, 'D2': 16, 'D3': 17, 'D4': 18, 'D5': 19,
        'E1': 20, 'E2': 21, 'E3': 22, 'E4': 23, 'E5': 24, 'F1': 25, 'F2': 26, 'F3': 27, 'F4': 28, 'F5': 29,
        'G1': 30, 'G2': 31, 'G3': 32, 'G4': 33, 'G5': 34
    }
    homeownership_mapping = {'MORTGAGE': 0, 'OWN': 1, 'RENT': 2}
    initial_list_status_mapping = {'f': 0, 'w': 1}
    verification_status_mapping = {'Not Verified': 0, 'Verified': 1}

    input_data[3] = grade_mapping[input_data[3]]
    input_data[4] = subgrade_mapping[input_data[4]]
    input_data[5] = homeownership_mapping[input_data[5]]
    input_data[7] = verification_status_mapping[input_data[7]]
    input_data[11] = initial_list_status_mapping[input_data[11]]

    return np.array(input_data).reshape(1, -1)

st.title("Loan Prediction Model")

amount = st.number_input("Amount")
funded_amount = st.number_input("Funded Amount Investor")
interest_rate = st.number_input("Interest Rate")

grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
sub_grade = st.selectbox("Sub Grade", [
    'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
    'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5',
    'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
    'G1', 'G2', 'G3', 'G4', 'G5'
])

home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT"])
salary = st.number_input("Salary")
verification_status = st.selectbox("Verification Status", ["Not Verified", "Verified"])
debit_to_income = st.number_input("Debit to Income")
open_account = st.number_input("Open Account")
total_accounts = st.number_input("Total Accounts")
initial_list_status = st.selectbox("Initial List Status", ["f", "w"])
total_received_interest = st.number_input("Total Received Interest")
total_received_late_fee = st.number_input("Total Received Late Fee")
recoveries = st.number_input("Recoveries")
collection_recovery_fee = st.number_input("Collection Recovery Fee")
last_week_pay = st.number_input("Last week Pay")
total_collection_amount = st.number_input("Total Collection Amount")
balance = st.number_input("Balance")

if st.button("Predict"):
    input_data = [
        amount, funded_amount, interest_rate, grade, sub_grade, home_ownership, salary,
        verification_status, debit_to_income, open_account, total_accounts, initial_list_status,
        total_received_interest, total_received_late_fee, recoveries, collection_recovery_fee,
        last_week_pay, total_collection_amount, balance
    ]

    processed_data = preprocess_data(input_data)
    prediction = model.predict(processed_data)
    
    # Display the result
    if prediction[0] == 1:
        st.write(f"RESULT: This customer will default the loan")
    else:
        st.write(f"RESULT: This customer will not default the loan")
