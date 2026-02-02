import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score, train_test_split
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


df = pd.read_csv("Data/CW1_train.csv")

target = "outcome"
y = df[target]
X = df.drop(columns=[target])

categorical_cols = ["cut", "color", "clarity"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


preprocessor_ridge = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

preprocessor_tree = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])


"""
Models being used for feature selection and data interpretation are regression, random forest and XGBoost
"""
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
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])


#cross-validation
def cross_validate_model(name, model):
    print("\n" + "=" * 60)
    print(f"Cross Validating: {name}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    #RMSE
    rmse_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    #R2
    r2_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2"
    )

    print(f"Mean RMSE: {-rmse_scores.mean():.4f}")
    print(f"Std RMSE : {rmse_scores.std():.4f}")

    print(f"Mean R2  : {r2_scores.mean():.4f}")
    print(f"Std R2   : {r2_scores.std():.4f}")


def train_model(name, model):
    print("\n" + "=" * 60)
    print(f"Training Final Model: {name}")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"Final Test RMSE: {rmse:.4f}")
    print(f"Final Test R²  : {r2:.4f}")

    return model


#Permutation Importance
def run_permutation_importance(name, model):
    print("\nPermutation Importance:", name)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="r2"
    )

    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    print("\nTop 15 Features:")
    print(perm_df.head(15))

    perm_df.head(15).plot(
        kind="barh",
        x="feature",
        y="importance",
        figsize=(8, 6)
    )
    plt.title(f"Permutation Importance: {name}")
    plt.gca().invert_yaxis()
    plt.show()



#SHAP Values
def run_tree_shap(name, model):
    print("\nSHAP Values:", name)

    X_test_transformed = model.named_steps["preprocessor"].transform(X_test)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    core_model = model.named_steps["model"]

    explainer = shap.TreeExplainer(core_model)
    shap_values = explainer.shap_values(X_test_transformed)

    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names
    )


#Main Execution place
cross_validate_model("Ridge Regression", ridge_model)
cross_validate_model("Random Forest", rf_model)
cross_validate_model("XGBoost", xgb_model)


ridge_model = train_model("Ridge Regression", ridge_model)
run_permutation_importance("Ridge Regression", ridge_model)


xgb_model = train_model("XGBoost", xgb_model)
run_permutation_importance("XGBoost", xgb_model)
run_tree_shap("XGBoost", xgb_model)

rf_model = train_model("Random Forest", rf_model)
run_permutation_importance("Random Forest", rf_model)
run_tree_shap("Random Forest", rf_model)
