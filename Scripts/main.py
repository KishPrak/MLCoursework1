import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pickle


df = pd.read_csv('Data/CW1_train.csv')
target = "outcome"

y = df[target]
features_top8 = ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'y', 'price']
X = df[features_top8]

categorical_cols = ["cut", "color", "clarity"]


def get_pipeline(model_type, features):
    current_cat = [c for c in categorical_cols if c in features]
    current_num = [c for c in features if c not in categorical_cols]

    if model_type in ['ridge', 'mlp', 'svr', 'hybrid']:
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), current_cat),
            ("num", StandardScaler(), current_num)
        ])
    else:
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), current_cat),
            ("num", "passthrough", current_num)
        ])

    if model_type == 'hybrid':
        nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42)
        rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
        model = VotingRegressor(
            estimators=[('nn', nn_model), ('rf', rf_model)],
            weights=[1, 1]
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


pipe = get_pipeline("hybrid", features_top8)

cv_scores = cross_val_score(
    pipe,
    X,
    y,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("=" * 55)
print("Hybrid Model (Top 8 Features)")
print(f"Mean R²: {cv_scores.mean():.4f}")
print(f"Std Dev: {cv_scores.std():.4f}")
print("=" * 55)


pipe.fit(X, y)

with open("hybrid_pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)

