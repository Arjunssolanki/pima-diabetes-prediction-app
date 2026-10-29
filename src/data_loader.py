import os
import yaml
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def ingest_data_from_s3():
    config = load_config()
    
    bucket_name = config["data"]["s3_bucket"]
    s3_key = config["data"]["s3_key"]
    raw_path = config["data"]["raw_path"]
    columns = config["data"]["columns"]

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"Connecting to AWS S3 bucket: {bucket_name}...")
        
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        
        print(f"Downloading {s3_key} from S3...")
        s3.download_file(bucket_name, s3_key, raw_path)
        print(f"Dataset successfully saved locally to {raw_path}")
    else:
        print(f"Dataset already exists locally at {raw_path}")

    df = pd.read_csv(raw_path, names=columns)
    return df


if __name__ == "__main__":
    df = ingest_data_from_s3()
    print(f"Data successfully verified. Ingested shape: {df.shape}")
