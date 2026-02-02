import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor


# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
train_df = pd.read_csv('Data/CW1_train.csv')
test_df = pd.read_csv('Data/CW1_test.csv')

target = "outcome"
# Use the Top 8 features + the engineered Depth squared
top_features = ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'y', 'price']
categorical_cols = ["cut", "color", "clarity"]
numeric_cols = [c for c in top_features if c not in categorical_cols]

def preprocess_with_engineered_features(data):
    # Apply the Depth^2 transformation
    data_copy = data.copy()
    data_copy['depth_sq'] = data_copy['depth'] ** 2
    return data_copy

# Prepare Train Data
X_train = preprocess_with_engineered_features(train_df[top_features])
y_train = train_df[target]
X_test_final = preprocess_with_engineered_features(test_df[top_features])

# Update numeric list to include the new feature
final_numeric = numeric_cols + ['depth_sq']

# ==========================================
# 2. DEFINE FINAL PIPELINE
# ==========================================
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), final_numeric)
])

# Use the parameters that yielded your best CV scores
nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    alpha=0.01, 
    early_stopping=True, 
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=300, 
    random_state=42, 
    n_jobs=-1
)

hybrid_voting = VotingRegressor(
    estimators=[('nn', nn_model), ('rf', rf_model)],
    weights=[1, 1]
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", hybrid_voting)
])

# ==========================================
# 3. TRAIN, SAVE, AND PREDICT
# ==========================================
print("Training final hybrid model...")
final_pipeline.fit(X_train, y_train)

# Save the model for your code supplement
joblib.dump(final_pipeline, 'final_hybrid_model.pkl')
print("Model saved as final_hybrid_model.pkl")

# Generate Predictions
print("Generating test set predictions...")
predictions = final_pipeline.predict(X_test_final)

# Save predictions to CSV (ensure the format matches project requirements)
output = pd.DataFrame({'outcome_pred': predictions})
output.to_csv('CW1_test_predictions.csv', index=False)
print("Predictions saved to CW1_test_predictions.csv")