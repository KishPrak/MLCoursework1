import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import shap
import matplotlib.pyplot as plt
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('Data/CW1_train.csv')
target = "outcome"

y = df[target]
X = df.drop(columns=[target])

categorical_cols = ["cut", "color", "clarity"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. PREPROCESSORS
# ============================================================

# Ridge needs scaling
preprocessor_ridge = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# Trees do not need scaling
preprocessor_tree = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])


# ============================================================
# 4. MODELS
# ============================================================

ridge_model = Pipeline([
    ("preprocessor", preprocessor_ridge),
    ("model", Ridge(alpha=1.0))
])

rf_model = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_model = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])


# ============================================================
# 5. TRAIN + EVALUATE
# ============================================================

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("=" * 55)
    print(f"{name} Results")
    print("RMSE:", rmse)
    print("R²:", r2)

    return model


# ============================================================
# 6. PERMUTATION IMPORTANCE FUNCTION
# ============================================================

def run_permutation_importance(name, model):
    print(f"\nPermutation Importance for {name}")

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="r2"
    )

    # IMPORTANT: permutation importance is on raw input features
    feature_names = X_test.columns

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    print("\nTop 15 Important Features:")
    print(perm_df.head(15))

    # Plot top 15
    perm_df.head(15).plot(
        kind="barh",
        x="feature",
        y="importance",
        figsize=(8, 6)
    )
    plt.title(f"Permutation Importance: {name}")
    plt.gca().invert_yaxis()
    plt.show()


# ============================================================
# 7. SHAP FUNCTION FOR TREE MODELS
# ============================================================

def run_tree_shap(name, model):
    print(f"\nSHAP Values for {name}")

    # Extract transformed features
    X_test_transformed = model.named_steps["preprocessor"].transform(X_test)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # Extract core model
    core_model = model.named_steps["model"]

    # SHAP explainer
    explainer = shap.TreeExplainer(core_model)
    shap_values = explainer.shap_values(X_test_transformed)

    # Summary plot
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names
    )


# ============================================================
# 8. SHAP FUNCTION FOR RIDGE (LINEAR SHAP)
# ============================================================

def run_ridge_shap(model):
    print("\nSHAP Values for Ridge Regression")

    # Transform features
    X_test_transformed = model.named_steps["preprocessor"].transform(X_test)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    ridge_core = model.named_steps["model"]

    # LinearExplainer works well for linear models
    explainer = shap.LinearExplainer(ridge_core, X_test_transformed)
    shap_values = explainer.shap_values(X_test_transformed)

    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names
    )


# ============================================================
# 9. RUN EVERYTHING
# ============================================================
"""
ridge_model = evaluate_model("Ridge Regression", ridge_model)
run_permutation_importance("Ridge Regression", ridge_model)
run_ridge_shap(ridge_model)
"""
rf_model = evaluate_model("Random Forest", rf_model)
#run_permutation_importance("Random Forest", rf_model)
run_tree_shap("Random Forest", rf_model)

"""
xgb_model = evaluate_model("XGBoost", xgb_model)
run_permutation_importance("XGBoost", xgb_model)
run_tree_shap("XGBoost", xgb_model)
"""