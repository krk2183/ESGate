import pandas as pd
import numpy as np
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import math, traceback, bcrypt,jwt,datetime,re,requests,json,torch,sqlite3, joblib

# -------------------------- Flask Setup --------------------------
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = "YOUR_SECRET_KEY_HERE"  # replace with env variable in prod

device = torch.device('cpu')  # force CPU for now

# -------------------------- DB Setup --------------------------
def init_db():
    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            company_name TEXT,
            int_rate REAL,
            default_rate REAL,
            sus_score REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -------------------------- JWT Helper --------------------------
from functools import wraps

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"message": "Token is missing!"}), 401
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['sub']
            role = data['role']
        except Exception as e:
            return jsonify({"message": f"Token invalid: {str(e)}"}), 401
        return f(current_user, role, *args, **kwargs)
    return decorated

# -------------------------- Login & Sign-up --------------------------
@app.route('/sign-up', methods=['POST'])
def signup_submit():
    username = request.json.get('username')
    password = request.json.get('password')
    role = request.json.get('role', 'user')
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return jsonify({"message": "Username already exists"}), 400
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed_password, role))
    conn.commit()
    conn.close()
    return jsonify({"message": "Sign-up successful"}), 200

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    
    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT password, role, id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
        role = result[1]
        user_id = result[2]
        payload = {
            'sub': username,
            'role': role,
            'id': user_id,
            'iat': datetime.datetime.utcnow(),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=12)
        }
        token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({"token": token, "role": role}), 200
    return jsonify({"message": "Invalid credentials"}), 401

class NeuralNet(nn.Module):
    """Generic PyTorch Neural Network Model for Classification or Regression."""
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2, is_classification=False):
        super(NeuralNet, self).__init__()
        self.is_classification = is_classification
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout)
            ])
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        out = torch.sigmoid(self.out(x))
        return out

# -------------------------- Prediction Models Setup --------------------------
scaler = joblib.load('assets/scaler.pkl')
model_reg_loaded = NeuralNet(input_size=25, hidden_sizes=[256,128,64], output_size=1).to(device)
model_reg_loaded.load_state_dict(torch.load("assets/interest_rate_prediction_model_best.pth", map_location=device))

# Residual MLP for default prediction
model = ResidualMLP(48).to(device)
model.load_state_dict(torch.load("assets/residual_mlp_sme.pth", map_location=device))
preprocessor = joblib.load("assets/preprocessor_sme_advanced.joblib")

income_bins = [4000.0, 35000.0, 49000.0, 63000.0, 86000.0, 6000000.0]
derived_feature_info = {
     'management_team_experience_median': 4.0,
     'income_bins': income_bins,
}
feature_names_order_reg = ['annual_revenue',
                        'office_ownership_status',
                        'management_team_experience',
                        'loan_amount',
                        'loan_percent_revenue',
                        'default_history',
                        'credit_history_length',
                        'years_in_operation_sqrt',
                        'debt_to_revenue_ratio',
                        'loan_to_operating_years_ratio_sqrt',
                        'loan_to_credit_hist_ratio',
                        'management_turnover_fraction_sqrt',
                        'log_revenue',
                        'log_loan_amount',
                        'op_years_sqrt_x_team_exp',
                        'company_size_group_Startup (0-5)',
                        'company_size_group_Growth (6-15)',
                        'company_size_group_Mature (16-30)',
                        'company_size_group_Established (31-50)',
                        'company_size_group_Legacy (51+)',
                        'revenue_bracket_1',
                        'revenue_bracket_2',
                        'revenue_bracket_3',
                        'revenue_bracket_4',
                        'revenue_bracket_5']


numerical_cols_to_scale_fit = ['annual_revenue',
                            'office_ownership_status',
                            'management_team_experience',
                            'loan_amount',
                            'loan_percent_revenue',
                            'default_history',
                            'credit_history_length',
                            'repayment_status',
                            'years_in_operation_sqrt',
                            'debt_to_revenue_ratio',
                            'loan_to_operating_years_ratio_sqrt',
                            'loan_to_credit_hist_ratio',
                            'management_turnover_fraction_sqrt',
                            'log_revenue',
                            'log_loan_amount',
                            'op_years_sqrt_x_team_exp'
                            ]

# --- Mistral AI setup ---
MISTRAL_API_KEY = "sk-or-v1-abcdef"
MISTRAL_API_URL = "https://openrouter.ai/api/v1/completions"
HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

# -------------------------- Useless Jargon --------------------------
def is_finite(num):
    try:
        return (num is not None) and np.isfinite(float(num))
    except Exception:
        return False
    
def validate_prediction_value(val,val_name='prediction',min_value=None,max_val=None):
    if val is None:
        return False, f"{val_name} is None"
    if not is_finite(val):
        return False, f"{val_name} is not finite: {val}"
    v = float(val)
    if (min_value is not None and v < min_value) or (max_val is not None and v> max_val):
        return False, f"{val_name} out of expected range [{min_value}, {max_val}]: {v}"
    return True, None

# predict_company_metrics, predict_default_sme, calculate_sustainability_score
def predict_company_metrics(model, scaler, derived_feature_info, feature_names_order, numerical_cols_to_scale_fit, device, metric_type,
                            years_in_operation, annual_revenue, office_ownership_status, management_team_experience, loan_amount,
                            default_history, credit_history_length, repayment_status):

    feature_values = {
        'years_in_operation': years_in_operation,
        'annual_revenue': annual_revenue,
        'office_ownership_status': office_ownership_status,
        'management_team_experience': management_team_experience,
        'loan_amount': loan_amount,
        'default_history': default_history,
        'credit_history_length': credit_history_length,
        'repayment_status': repayment_status  # Keep this - it's needed for scaling
    }
    user_df = pd.DataFrame([feature_values])

    # Derived features
    user_df['years_in_operation_sqrt'] = np.sqrt(user_df['years_in_operation'])
    years_in_op_sqrt = user_df['years_in_operation_sqrt'].iloc[0]
    user_df['debt_to_revenue_ratio'] = np.where(user_df['annual_revenue'] > 0, user_df['loan_amount'] / user_df['annual_revenue'], 0)
    user_df['loan_to_operating_years_ratio_sqrt'] = np.where(user_df['years_in_operation_sqrt'] > 0, user_df['loan_amount'] / user_df['years_in_operation_sqrt'], 0)
    user_df['loan_to_credit_hist_ratio'] = np.where(user_df['credit_history_length'] > 0, user_df['loan_amount'] / user_df['credit_history_length'], 0)
    user_df['management_turnover_fraction_sqrt'] = np.where(years_in_op_sqrt > 0, user_df['management_team_experience'] / years_in_op_sqrt, 0)
    user_df['log_revenue'] = np.log1p(user_df['annual_revenue'])
    user_df['log_loan_amount'] = np.log1p(user_df['loan_amount'])
    emp_length_imputed_user = user_df['management_team_experience'].fillna(derived_feature_info.get('management_team_experience_median', 0))
    user_df['op_years_sqrt_x_team_exp'] = user_df['years_in_operation_sqrt'] * emp_length_imputed_user

    # Add loan_percent_revenue which is also in the feature list
    user_df['loan_percent_revenue'] = np.where(user_df['annual_revenue'] > 0, 
                                                (user_df['loan_amount'] / user_df['annual_revenue']) * 100, 
                                                0)

    # Categoricals
    company_age_bins = [0, 5, 15, 30, 50, np.inf]
    company_age_labels = ['Startup (0-5)', 'Growth (6-15)', 'Mature (16-30)', 'Established (31-50)', 'Legacy (51+)']
    user_df['company_size_group'] = pd.cut(user_df['years_in_operation'], bins=company_age_bins, labels=company_age_labels, right=True, include_lowest=True)
    user_df['revenue_bracket'] = pd.cut(user_df['annual_revenue'], bins=derived_feature_info['income_bins'], labels=[1, 2, 3, 4, 5], include_lowest=True)
    user_df = pd.get_dummies(user_df, columns=['company_size_group'], prefix='company_size_group')
    user_df['revenue_bracket'] = user_df['revenue_bracket'].astype('category')
    user_df = pd.get_dummies(user_df, columns=['revenue_bracket'], prefix='revenue_bracket')

    # Drop years_in_operation AFTER creating all derived features
    user_df.drop('years_in_operation', axis=1, inplace=True)
    
    # DON'T reindex yet - we need to keep repayment_status for scaling!
    # First, ensure all numerical columns exist
    user_df = ensure_numeric_columns(user_df, numerical_cols_to_scale_fit)
    
    # Scale only the numerical columns (this includes repayment_status)
    scaled_values = scaler.transform(user_df[numerical_cols_to_scale_fit].values)
    user_df[numerical_cols_to_scale_fit] = scaled_values    
    # NOW drop repayment_status since it's not in the final feature set
    user_df.drop('repayment_status', axis=1, inplace=True, errors='ignore')
    
    # Reindex to match expected feature order (without repayment_status)
    user_df = user_df.reindex(columns=feature_names_order, fill_value=0)
    user_df = user_df.fillna(0).astype(np.float32)
    
    # Convert to tensor for model input
    user_input_tensor = torch.tensor(user_df.values, dtype=torch.float32).to(device)

    # In predict_company_metrics, before calling model:
    print(f"DataFrame shape before model: {user_df.shape}")
    print(f"DataFrame columns: {list(user_df.columns)}")
    print(f"Sample values:\n{user_df.iloc[0]}")

    model.eval()
    with torch.no_grad():
        output = model(user_input_tensor)
        if metric_type == 'rate':
            prediction = torch.expm1(output)
        elif metric_type == 'default':
            prediction = torch.sigmoid(output)
        else:
            raise ValueError("Invalid metric_type specified. Use 'rate' or 'default'.")
        
    # After model prediction:
    print(f"Raw model output: {output.item()}")
    print(f"After expm1: {prediction.item()}")
    return prediction.item()


def build_prompt(metrics_dict): # The model uses this prompt to generate standardized answers
    return f"""
        You are a financial and ESG advisor AI. A company has the following metrics:

        Interest Rate: {metrics_dict.get('int_rate', 'N/A')}
        Default Probability: {metrics_dict.get('default_rate', 'N/A')}
        Sustainability Score: {metrics_dict.get('sus_score', 'N/A')}
        Additional Notes: {metrics_dict.get('notes', '')}

        Task:
        1. Provide a concise summary (2-3 sentences) of the company's financial and sustainability health.
        2. Highlight strengths and weaknesses in separate bullet points.
        3. Suggest 2-3 actionable recommendations to improve financial or sustainability performance.

        Return the result in strict JSON format:
        {{
        "summary": "...",
        "strengths": ["..."],
        "weaknesses": ["..."],
        "recommendations": ["..."]
        }}
        """
def parse_mistral_output(response_json):
    try:
        output_text = response_json["choices"][0]["text"]
        # Remove ```json code blocks if present
        match = re.search(r"```json\s*(\{.*\})\s*```", output_text, re.DOTALL)
        if match:
            output_text = match.group(1)
        return json.loads(output_text)
    except Exception as e:
        return {"error": f"Failed to parse Mistral output: {str(e)}", "raw_text": response_json}


def ensure_numeric_columns(df, expected_cols):
    """Ensure all numeric columns exist in df, fill with 0 if missing."""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df

def predict_default_sme(sample_dict):
    df = pd.DataFrame([sample_dict])

    # --- Feature engineering ---
    massive_features = ["Annual Income", "Maximum Open Credit", "Current Loan Amount",
                        "Current Credit Balance", "Monthly Debt"]
    for col in massive_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    df["Debt_to_Income_Ratio"] = df["Monthly Debt"] / (df["Annual Income"] + 1e-6)
    df["Credit_Utilization"] = df["Current Credit Balance"] / (df["Maximum Open Credit"] + 1e-6)
    df["Loan_to_Income_Ratio"] = df["Current Loan Amount"] / (df["Annual Income"] + 1e-6)
    df["Credit_Problem_Score"] = (
        df["Number of Credit Problems"] +
        df["Bankruptcies"] * 2 +
        (df["Tax Liens"] > 0).astype(int)
    )

    # --- Ensure all columns from training exist ---
    numeric_cols_fitted = preprocessor.transformers_[0][2]
    categorical_cols_fitted = preprocessor.transformers_[1][2]

    for col in numeric_cols_fitted + categorical_cols_fitted:
        if col not in df.columns:
            df[col] = 0 if col in numeric_cols_fitted else "Unknown"

    # 🧩 Added fix: Ensure the transformed output always matches model’s 48 expected features
    expected_feature_order = [
        'num__Annual Income', 'num__Tax Liens', 'num__Number of Open Accounts',
        'num__Years of Credit History', 'num__Maximum Open Credit',
        'num__Number of Credit Problems', 'num__Months since last delinquent',
        'num__Bankruptcies', 'num__Current Loan Amount',
        'num__Current Credit Balance', 'num__Monthly Debt', 'num__Credit Score',
        'num__Debt_to_Income_Ratio', 'num__Credit_Utilization',
        'num__Loan_to_Income_Ratio', 'num__Credit_Problem_Score',
        'cat__Home Ownership_Have Mortgage', 'cat__Home Ownership_Home Mortgage',
        'cat__Home Ownership_Own Home', 'cat__Home Ownership_Rent',
        'cat__Years in current job_1 year', 'cat__Years in current job_10+ years',
        'cat__Years in current job_2 years', 'cat__Years in current job_3 years',
        'cat__Years in current job_4 years', 'cat__Years in current job_5 years',
        'cat__Years in current job_6 years', 'cat__Years in current job_7 years',
        'cat__Years in current job_8 years', 'cat__Years in current job_9 years',
        'cat__Years in current job_< 1 year', 'cat__Purpose_business loan',
        'cat__Purpose_buy a car', 'cat__Purpose_buy house',
        'cat__Purpose_debt consolidation', 'cat__Purpose_educational expenses',
        'cat__Purpose_home improvements', 'cat__Purpose_major purchase',
        'cat__Purpose_medical bills', 'cat__Purpose_moving', 'cat__Purpose_other',
        'cat__Purpose_renewable energy', 'cat__Purpose_small business',
        'cat__Purpose_take a trip', 'cat__Purpose_vacation',
        'cat__Purpose_wedding', 'cat__Term_Long Term', 'cat__Term_Short Term'
    ]

    # Transform input
    X_input = preprocessor.transform(df)

    # 🧩 Ensure correct shape (add missing columns as zeros if needed)
    X_df = pd.DataFrame(X_input, columns=preprocessor.get_feature_names_out())
    for col in expected_feature_order:
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df.reindex(columns=expected_feature_order, fill_value=0)

    # Convert to tensor
    X_input = torch.tensor(X_df.values, dtype=torch.float32).to(device)

    # --- Load model and predict ---
    model = ResidualMLP(X_input.shape[1]).to(device)
    model.load_state_dict(torch.load("assets/residual_mlp_sme.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(X_input).cpu().numpy()[0][0]

    return float(pred)




def calculate_sustainability_score(company_metrics, sector_averages, weights=None, tolerance=0.2, max_penalty=1.0):
    if weights is None:
        weights = {'energy_efficiency': 0.4, 'carbon_intensity': 0.3, 'water_usage': 0.3}

    normalized = {}
    for key in company_metrics:
        normalized[key] = company_metrics[key] / sector_averages[key]

    prelim_score = sum(weights[key] * normalized[key] for key in normalized) * 10 / sum(weights.values())

    penalty = 0
    for key in normalized:
        deviation = abs(normalized[key] - 1)  
        if deviation > tolerance:
            penalty += min(max_penalty, (deviation - tolerance) * max_penalty / (1 - tolerance))

    sustainability_score = max(0, prelim_score - penalty)  

    return [round(sustainability_score, 2),round(penalty,2)]

def get_mistral_summary(metrics_dict):
    payload = {
        "model": "mistralai/mistral-small-3.2-24b-instruct:free",
        "prompt": build_prompt(metrics_dict),
        "temperature": 0.7,
        "max_output_tokens": 500
    }
    response = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        return {"error": f"Mistral API error: {response.text}"}
    result = response.json()
    return parse_mistral_output(result)
# -------------------------- Prediction Endpoints --------------------------

# ---------- Safe predict_default ---------- :-(
@app.route('/predict_default', methods=['POST'])
@token_required
def predict_default(current_user, role):
    data_def = request.get_json()
    if not data_def:
        return jsonify({"error": "No JSON data received"}), 400

    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    if user_row is None:
        conn.close()
        return jsonify({"error": "User not found"}), 400
    user_id = user_row[0]

    try:
        default_rate = predict_default_sme(data_def)
    except Exception as e:
        traceback.print_exc()
        conn.close()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    ok, reason = validate_prediction_value(default_rate, "default_rate", 0.0, 1.0)
    if not ok:
        conn.close()
        return jsonify({"error": f"Invalid model output: {reason}"}), 500

    cursor.execute('''
        INSERT INTO predictions (user_id, company_name, default_rate)
        VALUES (?, ?, ?)
    ''', (user_id, data_def.get('company_name', ''), float(default_rate)))
    conn.commit()
    conn.close()

    return jsonify({'default_rate': float(default_rate)}), 200

# ---------- Safe predict_int_rate ----------
@app.route('/predict_int_rate', methods=['POST'])
@token_required
def predict_int_rate(current_user, role):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    if user_row is None:
        conn.close()
        return jsonify({"error": "User not found"}), 400
    user_id = user_row[0]

    required = ['operation_years','revenue','office_own','team_exp','loan_amt','default_hist','cred_hist_len','repayment_status']
    for k in required:
        if k not in data:
            conn.close()
            return jsonify({"error": f"Missing required field: {k}"}), 400

    try:
        # compute
        int_rate = predict_company_metrics(
            model_reg_loaded, scaler, derived_feature_info, feature_names_order_reg,
            numerical_cols_to_scale_fit, device, 'rate',
            data['operation_years'], data['revenue'], data['office_own'], data['team_exp'],
            data['loan_amt'], data['default_hist'], data['cred_hist_len'], data.get('repayment_status', 0)
        )
    except Exception as e:
        traceback.print_exc()
        conn.close()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Validate (choose sensible bounds)
    ok, reason = validate_prediction_value(int_rate, "int_rate", -0.5, 5.0)
    if not ok:
        # Do not insert invalid value into DB
        conn.close()
        return jsonify({"error": f"Invalid model output: {reason}"}), 500

    # Insert only valid value
    cursor.execute('''
        INSERT INTO predictions (user_id, company_name, int_rate)
        VALUES (?, ?, ?)
    ''', (user_id, data.get('company_name', ''), float(int_rate)))
    conn.commit()
    conn.close()

    return jsonify({'int_rate': float(int_rate)}), 200


@app.route('/user_predictions/<company>', methods=['GET'])
@token_required
def user_predictions(current_user, role, company):
    try:
        conn = sqlite3.connect('assets/database.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT int_rate, default_rate, sus_score, notes, created_at
            FROM predictions
            JOIN users ON predictions.user_id = users.id
            WHERE users.username = ? AND company_name = ?
            ORDER BY created_at ASC
        ''', (current_user, company))
        rows = cursor.fetchall()
        conn.close()

        formatted = []
        for r in rows:
            formatted.append({
                "int_rate": None if r[0] is None else float(r[0]),
                "default_rate": None if r[1] is None else float(r[1]),
                "sus_score": None if r[2] is None else float(r[2]),
                "notes": r[3],
                "created_at": r[4]
            })
        return jsonify({"predictions": formatted}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/sustainability_prediction', methods=['POST'])
@token_required
def sustainability_prediction(current_user, role):
    data_sus = request.get_json()
    
    # Get user_id from database like other endpoints
    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    if user_row is None:
        conn.close()
        return jsonify({"error": "User not found"}), 400
    user_id = user_row[0]
    
    dict_sus = {
        'energy_efficiency': data_sus.get('energy_ef'),
        'carbon_intensity': data_sus.get('carbon_int'),
        'water_usage': data_sus.get('water_usg')
    }
    sector_averages = {
        'energy_efficiency': 4.5,
        'carbon_intensity': 1.0,
        'water_usage': 3500
    }
    
    try:
        sus_score = calculate_sustainability_score(dict_sus, sector_averages)[0]
    except Exception as e:
        conn.close()
        return jsonify({"error": f"Calculation failed: {str(e)}"}), 500
    
    # Validate
    ok, reason = validate_prediction_value(sus_score, "sus_score", 0.0, 100.0)
    if not ok:
        conn.close()
        return jsonify({"error": f"Invalid sus_score: {reason}"}), 500
    
    cursor.execute('''
        INSERT INTO predictions (user_id, company_name, sus_score) 
        VALUES (?, ?, ?)
    ''', (user_id, data_sus.get('company_name', ''), float(sus_score)))
    conn.commit()
    conn.close()
    
    return jsonify({'sus_score': float(sus_score)}), 200


# -------------------------- AI Commentary --------------------------
@app.route('/company_summary', methods=['POST'])
@token_required
def company_summary(current_user, role):
    data = request.get_json()
    metrics_dict = {
        "int_rate": data.get("int_rate"),
        "default_rate": data.get("default_rate"),
        "sus_score": data.get("sus_score"),
        "notes": data.get("notes", "")
    }
    summary_result = get_mistral_summary(metrics_dict)
    return jsonify({"mistral_summary": summary_result}), 200


@app.route('/predict_all_and_save', methods=['POST'])
@token_required
def predict_all_and_save(current_user, role):
    payload = request.get_json()
    int_rate = payload.get('int_rate')
    default_rate = payload.get('default_rate')
    sus_score = payload.get('sus_score')
    company_name = payload.get('company_name', '')

    # Basic validation
    if int_rate is not None:
        ok, reason = validate_prediction_value(int_rate, "int_rate", -0.5, 5.0)
        if not ok:
            return jsonify({"error": f"Invalid int_rate: {reason}"}), 400
    if default_rate is not None:
        ok, reason = validate_prediction_value(default_rate, "default_rate", 0.0, 1.0)
        if not ok:
            return jsonify({"error": f"Invalid default_rate: {reason}"}), 400
    if sus_score is not None:
        ok, reason = validate_prediction_value(sus_score, "sus_score", 0.0, 100.0)
        if not ok:
            return jsonify({"error": f"Invalid sus_score: {reason}"}), 400

    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    if user_row is None:
        conn.close()
        return jsonify({"error": "User not found"}), 400
    user_id = user_row[0]

    cursor.execute('''
        INSERT INTO predictions (user_id, company_name, int_rate, default_rate, sus_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, company_name, int_rate, default_rate, sus_score))
    conn.commit()
    conn.close()

    return jsonify({"saved": True}), 200


# -------------------------- Run Server --------------------------
if __name__ == '__main__':
    app.run(debug=True)
