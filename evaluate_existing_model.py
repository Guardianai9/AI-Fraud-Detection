
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ml" / "model" / "fraud_classifier.joblib"
DATA_PATH = ROOT / "data" / "raw" / "sms_spam_collection.csv"
OUTPUT_PATH = ROOT / "model_test_results"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check the CSV path and filename in your project."
        )

    # Load the existing trained model; do not retrain it.
    model = joblib.load(MODEL_PATH)
    print("Loaded model:", type(model).__name__)

    # Read the SMS spam dataset.
    df = pd.read_csv(
        DATA_PATH, encoding="latin-1", usecols=["v1", "v2"]
    )
    df.columns = ["label", "text"]
    df = df.dropna().drop_duplicates()
    df["target"] = (df["label"].str.lower() == "spam").astype(int)

    from sklearn.model_selection import train_test_split

    # Reproduce the expected held-out split.
    _, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    X = test["text"].astype(str).tolist()
    y = test["target"].to_numpy()

    start = time.perf_counter()
    predictions = model.predict(X)
    elapsed = time.perf_counter() - start

    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "test_samples": len(y),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y, probabilities))
            if len(set(y)) > 1 else None
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "prediction_time_seconds": elapsed,
        "average_latency_ms": elapsed / len(y) * 1000,
    }

    print("\n===== MODEL PERFORMANCE =====")
    print(json.dumps(metrics, indent=2))
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Spam/Fraud"],
        zero_division=0,
    ))

    OUTPUT_PATH.mkdir(exist_ok=True)

    with open(
        OUTPUT_PATH / "metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    pd.DataFrame({
        "text": X,
        "actual_label": y,
        "predicted_label": predictions,
        "fraud_probability": probabilities,
        "correct": y == predictions,
    }).to_csv(OUTPUT_PATH / "predictions.csv", index=False)

    print("\nReports saved to:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
