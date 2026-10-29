import os
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def perform_lasso_selection():
    config = load_config()
    processed_path = config["data"]["processed_path"]
    target = config["data"]["target"]

    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data missing at {processed_path}. Run preprocessing.py first.")

    df = pd.read_csv(processed_path)
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=config["model"]["random_state"])
    lasso.fit(X_scaled, y)

    feature_coefficients = pd.Series(lasso.coef_[0], index=X.columns)
    selected_features = feature_coefficients[feature_coefficients != 0].index.tolist()
    dropped_features = feature_coefficients[feature_coefficients == 0].index.tolist()

    print("LASSO Feature Selection Results:")
    print(f" - Selected Features ({len(selected_features)}): {selected_features}")
    print(f" - Dropped Features ({len(dropped_features)}): {dropped_features}")

    return selected_features


if __name__ == "__main__":
    perform_lasso_selection()
