import requests
import json

BASE_URL = "http://127.0.0.1:5000"

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
        "Credit Score": data.get("cred_score", 0),
        "company_name": data.get("company_name", "TestCompany")
    }

def signup(username, password):
    payload = {"username": username, "password": password, "role": "user"}
    r = requests.post(f"{BASE_URL}/sign-up", json=payload)
    return r.json()

def login(username, password):
    payload = {"username": username, "password": password}
    r = requests.post(f"{BASE_URL}/login", json=payload)
    resp = r.json()
    if "token" not in resp:
        raise Exception(f"Login failed: {resp}")
    return resp["token"]

def safe_request(method, endpoint, token=None, payload=None):
    """Wrapper to handle requests safely with better error reporting"""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        
        if method == "POST":
            r = requests.post(f"{BASE_URL}/{endpoint}", json=payload, headers=headers)
        else:  # GET
            r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers)
        
        # Check if response has content
        if r.status_code == 404:
            return {"error": "Endpoint not found (404)", "status_code": 404}
        
        if r.text.strip() == "":
            return {"error": "Empty response from server", "status_code": r.status_code}
        
        try:
            return r.json()
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON response: {r.text[:200]}", "status_code": r.status_code}
    
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to server. Is it running?"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def run_all_tests():
    username = "kenan_test"
    password = "StrongPassword123" # Random password istg idk

    # Signup
    print_section("1. Sign Up")
    signup_resp = signup(username, password)
    print(json.dumps(signup_resp, indent=2))

    # Login
    print_section("2. Login")
    try:
        token = login(username, password)
        print(f"✓ Login successful! Token obtained.")
    except Exception as e:
        print(f"✗ Login failed: {e}")
        return

    # Interest Rate Prediction
    print_section("3. Interest Rate Prediction")
    int_rate_payload = {
        "operation_years": 4,
        "revenue": 150000,
        "office_own": 2,
        "team_exp": 10,
        "loan_amt": 25000,
        "default_hist": 0,
        "cred_hist_len": 3,
        "repayment_status": 0,
        "company_name": "TestCompany"
    }
    int_rate_resp = safe_request("POST", "predict_int_rate", token, int_rate_payload)
    print(json.dumps(int_rate_resp, indent=2))
    
    if "error" in int_rate_resp:
        print("⚠️  Interest rate prediction failed")

    # Default  Rate Prediction
    print_section("4. Default Probability Prediction")
    default_data_short = {
        "revenue": 150000,
        "office_own":2,
        "num_acc": 5,
        "credit_hist": 3,
        "max_cred": 50000,
        "num_prob": 0,
        "last_delinquent": 12,
        "loan_amt": 25000,
        "balance": 2000,
        "monthly_debit": 1000,
        "cred_score": 720,
        "company_name": "TestCompany"
    }
    default_payload = map_to_sme_features(default_data_short)
    default_resp = safe_request("POST", "predict_default", token, default_payload)
    print(json.dumps(default_resp, indent=2))

    # Sustainability Prediction
    print_section("5. Sustainability Score Prediction")
    sustainability_payload = {
        "energy_ef": 4.2,
        "carbon_int": 1.1,
        "water_usg": 3400,
        "company_name": "TestCompany"
    }
    sus_resp = safe_request("POST", "sustainability_prediction", token, sustainability_payload)
    print(json.dumps(sus_resp, indent=2))

    # Company Summary (Mistral AI)
    print_section("6. AI-Generated Company Summary")
    summary_payload = {
        "int_rate": int_rate_resp.get("int_rate", 0.07),
        "default_rate": default_resp.get("default_rate", 0.12),
        "sus_score": sus_resp.get("sus_score", 7.5),
        "notes": "Company is small but growing fast."
    }
    summary_resp = safe_request("POST", "company_summary", token, summary_payload)
    print(json.dumps(summary_resp, indent=2))

    # Retrieve Past Predictions
    print_section("7. Past Predictions for TestCompany")
    past_preds = safe_request("GET", "user_predictions/TestCompany", token)
    print(json.dumps(past_preds, indent=2))

    #  Test Combined Save Endpoint
    print_section("8. Save Combined Prediction")
    combined_payload = {
        "int_rate": int_rate_resp.get("int_rate"),
        "default_rate": default_resp.get("default_rate"),
        "sus_score": sus_resp.get("sus_score"),
        "company_name": "TestCompany_Combined"
    }
    combined_resp = safe_request("POST", "predict_all_and_save", token, combined_payload)
    print(json.dumps(combined_resp, indent=2))

    # Test Edge Cases
    print_section("9. Edge Case Tests")
    
    # Test with missing required field
    print("Testing missing required field...")
    bad_payload = {"operation_years": 4, "revenue": 150000}  # Missing other fields
    bad_resp = safe_request("POST", "predict_int_rate", token, bad_payload)
    print(f"Missing field test: {bad_resp.get('error', 'Unexpected success')}\n")
    
    # Test with invalid token
    print("Testing invalid token...")
    invalid_resp = safe_request("POST", "predict_int_rate", "invalid_token", int_rate_payload)
    print(f"Invalid token test: {invalid_resp.get('message', 'No error message')}\n")

    print_section("✓ All Tests Complete")

if __name__ == "__main__":
    run_all_tests()