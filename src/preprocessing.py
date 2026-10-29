import os
import yaml
import numpy as np
import pandas as pd
from scipy import stats


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def check_normality(df, feature):
    skewness = df[feature].skew()
    stat, p_value = stats.shapiro(df[feature].dropna())
    is_normal = p_value > 0.05
    return {"skewness": skewness, "p_value": p_value, "is_normal": is_normal}


def apply_transformations(df, features_to_transform):
    transformed_df = df.copy()
    transformation_summary = {}

    for col in features_to_transform:
        if (df[col] <= 0).any():
            transformed_data, maxlog = stats.yeojohnson(df[col])
            method = "Yeo-Johnson"
        else:
            transformed_data, maxlog = stats.boxcox(df[col])
            method = "Box-Cox"

        transformed_df[col] = transformed_data
        
        post_stat, post_p = stats.shapiro(transformed_data)
        transformation_summary[col] = {
            "method": method,
            "original_skew": df[col].skew(),
            "transformed_skew": transformed_df[col].skew(),
            "is_normal_now": post_p > 0.05
        }

    return transformed_df, transformation_summary


def run_preprocessing_pipeline():
    config = load_config()
    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    target = config["data"]["target"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run data_loader.py first.")

    df = pd.read_csv(raw_path, names=config["data"]["columns"])
    feature_cols = [col for col in df.columns if col != target]

    skewed_features = []
    print("Initial Normality Report:")
    for col in feature_cols:
        report = check_normality(df, col)
        print(f" - {col}: Skewness={report['skewness']:.2f}, Normal={report['is_normal']}")
        if not report["is_normal"] or abs(report["skewness"]) > 0.5:
            skewed_features.append(col)

    processed_df, summary = apply_transformations(df, skewed_features)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    print(f"\nTransformed datasets saved to {processed_path}")
    
    return summary


if __name__ == "__main__":
    run_preprocessing_pipeline()
