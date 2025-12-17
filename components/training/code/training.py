import argparse
import os
import json
import joblib
import pandas as pd
import mlflow

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def log_classification_report(report: dict, prefix: str = "classification_report"):
    """
    Logs metrics from sklearn classification_report(output_dict=True)
    to MLflow as flat metric keys.
    """
    for k, v in report.items():
        if isinstance(v, dict):
            for m, val in v.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"{prefix}.{k}.{m}", float(val))
        elif isinstance(v, (int, float)):
            mlflow.log_metric(f"{prefix}.{k}", float(v))


def main():
    parser = argparse.ArgumentParser()

    # Inputs (folders coming from previous component)
    parser.add_argument("--train_ready", type=str, required=True)
    parser.add_argument("--test_ready", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="house_affiliation")

    # Hyperparams
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)

    # Outputs (AzureML uri_folder outputs)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)

    args = parser.parse_args()

    # Expected files (created by prepare/split step)
    X_train_path = os.path.join(args.train_ready, "X_train.csv")
    y_train_path = os.path.join(args.train_ready, "y_train.csv")
    X_test_path = os.path.join(args.test_ready, "X_test.csv")
    y_test_path = os.path.join(args.test_ready, "y_test.csv")

    for p in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # Load CSVs
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)[args.target_col].astype(str)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)[args.target_col].astype(str)

    # Train model
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # MLflow logging (AzureML job usually auto-configures tracking)
    mlflow.log_param("target_col", args.target_col)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
    mlflow.log_param("random_state", args.random_state)

    mlflow.log_metric("accuracy", acc)
    log_classification_report(report, prefix="classification_report")

    # Save model to AzureML output folder
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.joblib")
    joblib.dump(clf, model_path)

    # ✅ NEW: save the feature columns used during training (needed for deployment one-hot alignment)
    feature_cols_path = os.path.join(args.model_output, "feature_columns.json")
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f, indent=2)

    print(f"✅ Saved model to: {model_path}")
    print(f"✅ Saved feature columns to: {feature_cols_path}")

    # Save metrics JSON to AzureML output folder
    os.makedirs(args.metrics_output, exist_ok=True)
    metrics_path = os.path.join(args.metrics_output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "classification_report": report}, f, indent=2)

    print(f"✅ Saved metrics to: {metrics_path}")
    print("✅ Training done")
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
