import pickle
import pandas as pd

MODEL_PATH = "Model/hybrid_pipeline.pkl"
INPUT_CSV = "Data/CW1_test.csv"
OUTPUT_CSV = "predictions.csv"
TARGET = "outcome"
FEATURES_TOP8 = ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'y', 'price']


with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

df = pd.read_csv(INPUT_CSV)
X = df[FEATURES_TOP8]


if TARGET in df.columns:
    y_true = df[TARGET]
else:
    y_true = None

preds = pipeline.predict(X)

out_df = pd.DataFrame({
    "prediction": preds
})

if y_true is not None:
    out_df["actual"] = y_true.values
    out_df["error"] = out_df["prediction"] - out_df["actual"]

out_df.to_csv(OUTPUT_CSV, index=False)

print(f"Predictions written to {OUTPUT_CSV}")
