import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib


device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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
    

scaler = joblib.load('scaler.pkl')

numerical_cols_to_scale_fit = ['annual_revenue',
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
                            'op_years_sqrt_x_team_exp']

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
        'repayment_status': repayment_status
    }
    user_df = pd.DataFrame([feature_values])

    user_df['years_in_operation_sqrt'] = np.sqrt(user_df['years_in_operation'])
    years_in_op_sqrt = user_df['years_in_operation_sqrt'].iloc[0] 

    user_df['debt_to_revenue_ratio'] = user_df.apply(lambda row: row['loan_amount'] / row['annual_revenue'] if row['annual_revenue'] > 0 else 0, axis=1)
    user_df['loan_to_operating_years_ratio_sqrt'] = user_df.apply(lambda row: row['loan_amount'] / row['years_in_operation_sqrt'] if row['years_in_operation_sqrt'] > 0 else 0, axis=1)
    user_df['loan_to_credit_hist_ratio'] = user_df.apply(lambda row: row['loan_amount'] / row['credit_history_length'] if row['credit_history_length'] > 0 else 0, axis=1)
    user_df['management_turnover_fraction_sqrt'] = user_df.apply(lambda row: row['management_team_experience'] / years_in_op_sqrt if (pd.notna(row['management_team_experience']) and years_in_op_sqrt > 0) else 0, axis=1)
    user_df['log_revenue'] = np.log1p(user_df['annual_revenue'])
    user_df['log_loan_amount'] = np.log1p(user_df['loan_amount'])
    
    emp_length_imputed_user = user_df['management_team_experience'].fillna(derived_feature_info.get('management_team_experience_median', 0))
    user_df['op_years_sqrt_x_team_exp'] = user_df['years_in_operation_sqrt'] * emp_length_imputed_user

    company_age_bins = [0, 5, 15, 30, 50, np.inf]
    company_age_labels = ['Startup (0-5)', 'Growth (6-15)', 'Mature (16-30)', 'Established (31-50)', 'Legacy (51+)']
    user_df['company_size_group'] = pd.cut(user_df['years_in_operation'], bins=company_age_bins, labels=company_age_labels, right=True, include_lowest=True)
    user_df['revenue_bracket'] = pd.cut(user_df['annual_revenue'], bins=derived_feature_info['income_bins'], labels=[1, 2, 3, 4, 5], include_lowest=True)

    user_df = pd.get_dummies(user_df, columns=['company_size_group'], prefix='company_size_group')
    user_df['revenue_bracket'] = user_df['revenue_bracket'].astype('category')
    user_df = pd.get_dummies(user_df, columns=['revenue_bracket'], prefix='revenue_bracket')
    
    user_df.drop('years_in_operation', axis=1, inplace=True) 

    bool_cols_user = user_df.select_dtypes(include=[bool]).columns
    user_df[bool_cols_user] = user_df[bool_cols_user].astype(int)
    
    user_df = user_df.reindex(columns=feature_names_order, fill_value=0)
    
    user_df = user_df.astype(np.float32) 
    
    user_df.fillna(0, inplace=True)

    cols_to_scale_for_prediction = [col for col in numerical_cols_to_scale_fit if col in user_df.columns]
    user_df[cols_to_scale_for_prediction] = scaler.transform(user_df[cols_to_scale_for_prediction])
    
    user_input_tensor = torch.tensor(user_df.values, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        output = model(user_input_tensor)

        if metric_type == 'rate':
            prediction = torch.expm1(output)
        elif metric_type == 'default':
            prediction = torch.sigmoid(output)
        else:
            raise ValueError("Invalid metric_type specified. Use 'rate' or 'default'.")

    return prediction.item()

model_reg_loaded = NeuralNet(input_size=25, hidden_sizes=[256, 128, 64], output_size=1).to(device)

model_reg_loaded.load_state_dict(torch.load("interest_rate_prediction_model_best.pth"))

# Test variables
years_in_op=4
annual_rev=150000
office_own=0 #  {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3, 'ANY': 4}
team_exp=10
loan_amt=25000
default_hist=0
cred_hist_len=3
repay_status=0
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

predicted_rate_no_grade = predict_company_metrics(
    model_reg_loaded, scaler, derived_feature_info, feature_names_order_reg, # Use REG features
    numerical_cols_to_scale_fit, device, 'rate', 
    years_in_op, annual_rev, office_own, team_exp, loan_amt, default_hist, cred_hist_len, repay_status
)

# INT_RATE PREDICTION

print(f"\n--- Prediction Results for a 4-year-old Startup with $150k Revenue ---")
# print(f"Predicted Default Probability (Model 1): {predicted_default_prob * 100:.2f}%")
print(f"Predicted Interest Rate (Model 2, No Credit Grade): {predicted_rate_no_grade:.3f}%")



# DEFAULTING PREDICTION
# New model will be implemented
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

model = ResidualMLP(48)
state_dict = torch.load('residual_mlp_sme.pth')
model.load_state_dict(state_dict)

preprocessor = joblib.load('preprocessor_sme_advanced.joblib')

# Prediction algorithm for the default prediction
def predict_default_sme(sample_dict):
    df = pd.DataFrame([sample_dict])
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

    # Preprocess
    preprocessor = joblib.load("preprocessor_sme_advanced.joblib")
    X_input = preprocessor.transform(df)
    X_input = torch.tensor(X_input, dtype=torch.float32).to(device)

    # Load model
    model = ResidualMLP(X_input.shape[1]).to(device)
    model.load_state_dict(torch.load("residual_mlp_sme.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(X_input).cpu().numpy()[0][0]
    return float(pred)



def calculate_sustainability_score(company_metrics, sector_averages, weights=None, tolerance=0.2, max_penalty=1.0):
    """
    Calculate a Sustainability Score for a company based on Energy Efficiency, Carbon Intensity, and Water Usage.

    Parameters:
    ----->   - company_metrics: dict with keys 'energy_efficiency', 'carbon_intensity', 'water_usage'  <-------
    - sector_averages: dict with same keys representing sector average metrics
    - weights: dict with keys 'energy_efficiency', 'carbon_intensity', 'water_usage' (default equal weights)
    - tolerance: float, acceptable deviation from sector average before penalty (default 20%)
    - max_penalty: maximum penalty points applied for extreme deviation (default 0.5)

    Returns:
    - sustainability_score: float, final score between 0 and 10
    """

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


company_metrics = {
    'energy_efficiency': 0.2,  # revenue per MWh
    'carbon_intensity': 1.3,   # tCO2e per revenue unit
    'water_usage': 3000         # m³ per revenue unit
}

sector_averages = {
    'energy_efficiency': 4.5,
    'carbon_intensity': 1.0,
    'water_usage': 3500
}

score = calculate_sustainability_score(company_metrics, sector_averages)
print("Sustainability Score:", score)


