import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_and_evaluate_models():
    config = load_config()
    processed_path = config["data"]["processed_path"]
    target = config["data"]["target"]
    seed = config["model"]["random_state"]
    test_size = config["model"]["test_size"]

    df = pd.read_csv(processed_path)
    
    # Feature Engineering: Add interaction terms directly
    df["Glucose_Insulin_Ratio"] = df["Glucose"] * (df["Insulin"] + 1)
    df["Age_BMI_Factor"] = df["Age"] * df["BMI"]
    
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Apply SMOTE strictly on the training partition
    smote = SMOTE(random_state=seed)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        "RandomForest": RandomForestClassifier(random_state=seed),
        "SupportVectorMachine": SVC(random_state=seed, probability=True),
        "XGBoost": XGBClassifier(random_state=seed, eval_metric="logloss")
    }

    param_grids = {
        "RandomForest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [4, 6, 8, None],
            "classifier__min_samples_split": [2, 5, 10]
        },
        "SupportVectorMachine": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"]
        },
        "XGBoost": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__subsample": [0.8, 0.9]
        }
    }

    os.makedirs("models", exist_ok=True)
    print("Starting Multi-Model Hyperparameter Tuning with SMOTE and Pipelines...\n")

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid_search.fit(X_train_res, y_train_res)
        best_pipeline = grid_search.best_estimator_

        preds = best_pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print("========================================")
        print(f"Model: {name}")
        print(f" Best Params: {grid_search.best_params_}")
        print(f" Test Accuracy: {acc:.4f}")
        print(f" F1-Score: {f1:.4f}\n")

        joblib.dump(best_pipeline, f"models/{name.lower()}_model.joblib")

    print("All model pipelines trained and serialized successfully in models/ directory!")


if __name__ == "__main__":
    train_and_evaluate_models()