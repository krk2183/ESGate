from flask import Flask,request
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2):
        super(Model, self).__init__()

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

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

hidden_sizes = [256,128, 64]
inp_size = 33 # X_train.shape[1]
out_size = 1
model = Model(inp_size, hidden_sizes, out_size)
model.load_state_dict(torch.load("complete_credit_score_model.pth"))
model.to(device)

def predict_loan_int_rate(model, scaler, derived_feature_info, feature_names_order, numerical_cols_to_scale_fit, device,
                            person_age, person_income, person_home_ownership, person_emp_length, loan_amnt,
                            cb_person_default_on_file, cb_person_cred_hist_length, loan_status, loan_grade):

    feature_values = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'loan_status': loan_status,
        'loan_grade': loan_grade
    }
    user_df = pd.DataFrame([feature_values])

    required_for_derived = ['person_income', 'person_age', 'cb_person_cred_hist_length', 'person_emp_length', 'loan_amnt']
    for col in required_for_derived:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df['loan_percent_income'] = user_df.apply(
        lambda row: row['loan_amnt'] / row['person_income'] if row['person_income'] > 0 else (
            derived_feature_info.get('loan_percent_income_default', 0)
        ),
        axis=1
    )

    user_df['loan_to_income_ratio'] = user_df.apply(
        lambda row: row['loan_amnt'] / row['person_income'] if row['person_income'] > 0 else (
            derived_feature_info.get('loan_to_income_ratio_default', 0)
        ),
        axis=1
    )

    user_df['loan_to_age_ratio'] = user_df.apply(
        lambda row: row['loan_amnt'] / row['person_age'] if row['person_age'] > 0 else (
            derived_feature_info.get('loan_to_age_ratio_default', 0)
        ),
        axis=1
    )

    user_df['loan_to_cred_hist_ratio'] = user_df.apply(
        lambda row: row['loan_amnt'] / row['cb_person_cred_hist_length'] if row['cb_person_cred_hist_length'] > 0 else (
            derived_feature_info.get('loan_to_cred_hist_ratio_default', 0)
        ),
        axis=1
    )

    user_df['emp_hist_fraction'] = user_df.apply(
        lambda row: row['person_emp_length'] / row['person_age'] if (
            pd.notna(row['person_emp_length']) and row['person_age'] > 0
        ) else (
             derived_feature_info.get('emp_hist_fraction_default', 0)
        ),
        axis=1
    )

    user_df['log_person_income'] = np.log1p(user_df['person_income'])
    user_df['log_loan_amnt'] = np.log1p(user_df['loan_amnt'])

    emp_length_imputed_user = user_df['person_emp_length'].fillna(derived_feature_info.get('person_emp_length_median', 0))
    user_df['age_x_emp_length'] = user_df['person_age'] * emp_length_imputed_user

    age_bins = [0, 25, 35, 45, 55, np.inf]
    age_labels = ['0-25', '26-35', '36-45', '46-55', '56+']
    user_df['age_group'] = pd.cut(user_df['person_age'], bins=age_bins, labels=age_labels, right=True, include_lowest=True)

    if 'income_bins' in derived_feature_info:
          try:
              user_df['income_bracket'] = pd.cut(user_df['person_income'], bins=derived_feature_info['income_bins'], labels=[1, 2, 3, 4, 5], include_lowest=True)
          except Exception as e:
              user_df['income_bracket'] = np.nan
    else:
          user_df['income_bracket'] = np.nan

    if 'person_home_ownership' in user_df.columns and user_df['person_home_ownership'].dtype == object and 'ownership_mapping' in derived_feature_info:
          user_df['person_home_ownership'] = user_df['person_home_ownership'].map(derived_feature_info['ownership_mapping'])

    if 'cb_person_default_on_file' in user_df.columns and user_df['cb_person_default_on_file'].dtype == object:
        user_df['cb_person_default_on_file'] = user_df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    for grade in range(7): # Grades 0 through 6
        col_name = f'loan_grade_{grade}'
        user_df[col_name] = 0

    if 'loan_grade' in user_df.columns and not user_df.empty:
        grade_value = int(user_df['loan_grade'].iloc[0])
        if 0 <= grade_value <= 6:
            user_df[f'loan_grade_{grade_value}'] = 1
        user_df.drop('loan_grade', axis=1, inplace=True)

    user_df = pd.get_dummies(user_df, columns=['age_group'], prefix='age_group')

    if 'income_bracket' in user_df.columns:
         user_df['income_bracket'] = user_df['income_bracket'].astype('category')
         user_df = pd.get_dummies(user_df, columns=['income_bracket'], prefix='income_bracket')

    user_df = user_df.reindex(columns=feature_names_order, fill_value=0)

    for feature in feature_names_order:
        user_df[feature] = user_df[feature].astype(np.float32)

    user_df.fillna(0, inplace=True)


    cols_to_scale_for_prediction = [col for col in numerical_cols_to_scale_fit if col in user_df.columns]

    user_df[cols_to_scale_for_prediction] = scaler.transform(user_df[cols_to_scale_for_prediction])

    user_input_tensor = torch.tensor(user_df.values, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(user_input_tensor)

    return prediction.item()

ownership_mapping = {
    "RENT": 0,
    "MORTGAGE": 1,
    "OWN":2
}

income_bins = [-1.0042542439643916, -0.4932516827369155, -0.2689712169141602, -0.04312089996286911, 0.32531354857104783, 95.04500459992109] # df['person_income'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).tolist()

# Define derived info
derived_info = {
     'ownership_mapping': ownership_mapping,
     'person_emp_length_median': 4.0, # dataset['person_emp_length'].median()
     'income_bins': income_bins,
}

scaler = joblib.load('standard_scaler_joblib.pkl')
grade_mapping = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}

numerical_cols_to_scale_fit = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_amnt', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_to_income_ratio', 'loan_to_age_ratio', 'loan_to_cred_hist_ratio', 'emp_hist_fraction', 'log_person_income', 'log_loan_amnt', 'age_x_emp_length']

feature_names_order = ['person_age',
 'person_income',
 'person_home_ownership',
 'person_emp_length',
 'loan_amnt',
 'loan_status',
 'loan_percent_income',
 'cb_person_default_on_file',
 'cb_person_cred_hist_length',
 'loan_to_income_ratio',
 'loan_to_age_ratio',
 'loan_to_cred_hist_ratio',
 'emp_hist_fraction',
 'log_person_income',
 'log_loan_amnt',
 'age_x_emp_length',
 'age_group_0-25',
 'age_group_26-35',
 'age_group_36-45',
 'age_group_46-55',
 'age_group_56+',
 'loan_grade_0',
 'loan_grade_1',
 'loan_grade_2',
 'loan_grade_3',
 'loan_grade_4',
 'loan_grade_5',
 'loan_grade_6',
 'income_bracket_1',
 'income_bracket_2',
 'income_bracket_3',
 'income_bracket_4',
 'income_bracket_5']

predicted_rate_from_args = predict_loan_int_rate(
    model,
    scaler,
    derived_info,
    feature_names_order,
    numerical_cols_to_scale_fit, # This is the corrected list of feature columns
    device,
    person_age=50,
    person_income=1500,
    person_home_ownership=ownership_mapping['MORTGAGE'], # Using the mapped value (1)
    person_emp_length=2,
    loan_amnt=1000,
    cb_person_default_on_file=0,
    cb_person_cred_hist_length=2,
    loan_status=1,
    loan_grade=grade_mapping['A'] # Using the mapped value (3)
)

print(f"\nPredicted loan interest rate using provided values: {predicted_rate_from_args:.3f}")
