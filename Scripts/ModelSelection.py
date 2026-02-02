import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor



df = pd.read_csv('Data/CW1_train.csv')
target = "outcome"
y = df[target]
features_full = df.drop(columns=[target]).columns.tolist()
features_top8 = ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'y', 'price']

categorical_cols = ["cut", "color", "clarity"]

def get_pipeline(model_type, features):
    current_cat = [c for c in categorical_cols if c in features]
    current_num = [c for c in features if c not in categorical_cols]
    
    if model_type in ['ridge', 'mlp', 'svr', 'hybrid']:
        proc = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), current_cat),
            ("num", StandardScaler(), current_num)
        ])
    else:
        proc = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), current_cat),
            ("num", "passthrough", current_num)
        ])
    
    #Model Selection
    if model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
    elif model_type == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=42)
    elif model_type == 'svr':
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    elif model_type == 'hybrid':
        nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
        model = VotingRegressor(
            estimators=[('nn', nn_model), ('rf', rf_model)],
            weights=[1, 1] 
        )
    
    pipeline = Pipeline([("preprocessor", proc), ("model", model)])
    return pipeline


results = []
feature_scenarios = [("Full (30)", features_full), ("Top 8", features_top8)]
models = [
    ('Ridge', 'ridge'), 
    ('Random Forest', 'rf'), 
    ('XGBoost', 'xgb'),
    ('Neural Network', 'mlp'),
    ('SVR', 'svr'),
    ('Hybrid', 'hybrid')
    
]
print(f"{'Model':<15} | {'Features':<10} | {'Mean R²':<10} | {'Std Dev':<10}")
print("-" * 55)

for f_name, f_list in feature_scenarios:
    X_subset = df[f_list]
    for m_label, m_type in models:
        pipe = get_pipeline(m_type, f_list)
        
        cv_scores = cross_val_score(pipe, X_subset, y, cv=5, scoring='r2')
        mean_r2 = cv_scores.mean()
        std_r2 = cv_scores.std()
        
        results.append({'Model': m_label, 'Set': f_name, 'R2': mean_r2})
        print(f"{m_label:<15} | {f_name:<10} | {mean_r2:.4f}     | {std_r2:.4f}")



res_df = pd.DataFrame(results)
res_df.pivot(index='Model', columns='Set', values='R2').plot(kind='bar', figsize=(10,6))
plt.ylabel("Cross-Validated R²")
plt.title("Impact of Feature Selection on Model Performance")
plt.show()

