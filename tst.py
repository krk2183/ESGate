import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# ------------------------------
# Helper to map short test keys to SME model features
# ------------------------------
def map_to_sme_features(data):
    return {
        "Home Ownership": "Own",
        "Annual Income": data.get("revenue", 0),
        "Years in current job": "5-10 years",
        "Tax Liens": 0,
        "Number of Open Accounts": data.get("num_acc", 0),
        "Years of Credit History": data.get("credit_hist", 0),
        "Maximum Open Credit": data.get("max_cred", 0),
        "Number of Credit Problems": data.get("num_prob", 0),
        "Months since last delinquent": data.get("last_delinquent", 0),
        "Bankruptcies": 0,
        "Purpose": "Expansion",
        "Term": "36 months",
        "Current Loan Amount": data.get("loan_amt", 0),
        "Current Credit Balance": data.get("balance", 0),
        "Monthly Debt": data.get("monthly_debit", 0),
        "Credit Score": data.get("cred_score", 0)
    }

# ------------------------------
# 1. Sign Up
# ------------------------------
signup_data = {
    "username": "kenan_test",
    "password": "StrongPassword123",
    "role": "user"
}

r = requests.post(f"{BASE_URL}/sign-up", json=signup_data)
print("Sign-up response:", r.json())

# ------------------------------
# 2. Login
# ------------------------------
login_data = {
    "username": "kenan_test",
    "password": "StrongPassword123"
}

r = requests.post(f"{BASE_URL}/login", json=login_data)
login_resp = r.json()
print("Login response:", login_resp)

if "token" not in login_resp:
    raise Exception("Login failed, cannot continue tests.")

token = login_resp["token"]
headers = {"Authorization": f"Bearer {token}"}

# ------------------------------
# 3. Interest Rate Prediction
# ------------------------------
int_rate_data = {
    "operation_years": 4,
    "revenue": 150000,
    "office_own": 0,
    "team_exp": 10,
    "loan_amt": 25000,
    "default_hist": 0,
    "cred_hist_len": 3,
    "repayment_status": 0,
    "user_id": 1,
    "company_name": "TestCompany"
}

r = requests.post(f"{BASE_URL}/predict_int_rate", json=int_rate_data, headers=headers)
print("Interest Rate Prediction:", r.json())

# ------------------------------
# 4. Default Prediction
# ------------------------------
default_data_short = {
    "revenue": 150000,
    "num_acc": 5,
    "credit_hist": 3,
    "max_cred": 50000,
    "num_prob": 0,
    "last_delinquent": 12,
    "loan_amt": 25000,
    "balance": 2000,
    "monthly_debit": 1000,
    "cred_score": 720,
    "user_id": 1,
    "company_name": "TestCompany"
}

# Map short keys to full model features
default_data = map_to_sme_features(default_data_short)
r = requests.post(f"{BASE_URL}/predict_default", json=default_data, headers=headers)
print("Default Prediction:", r.json())

# ------------------------------
# 5. Sustainability Prediction
# ------------------------------
sustainability_data = {
    "energy_ef": 4.2,
    "carbon_int": 1.1,
    "water_usg": 3400,
    "user_id": 1,
    "company_name": "TestCompany"
}

r = requests.post(f"{BASE_URL}/sustainability_prediction", json=sustainability_data, headers=headers)
print("Sustainability Prediction:", r.json())

# ------------------------------
# 6. AI Company Summary
# ------------------------------
summary_data = {
    "int_rate": 0.072,
    "default_rate": 0.12,
    "sus_score": 7.8,
    "notes": "Company is small but growing fast.",
    "user_id": 1,
    "company_name": "TestCompany"
}

r = requests.post(f"{BASE_URL}/company_summary", json=summary_data, headers=headers)
print("Mistral AI Summary:", json.dumps(r.json(), indent=4))

# ------------------------------
# 7. Retrieve Past Predictions
# ------------------------------
r = requests.get(f"{BASE_URL}/user_predictions/TestCompany", headers=headers)
print("Past Predictions:", json.dumps(r.json(), indent=4))
