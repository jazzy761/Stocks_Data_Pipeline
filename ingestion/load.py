from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import os 
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET    = os.getenv("BQ_DATASET")
TABLE      = os.getenv("BQ_TABLE")

def get_client():
    return bigquery.Client(project = PROJECT_ID)

def ensure_dataset_exists(client):
    dataset_ref = f"{PROJECT_ID}.{DATASET}"
    
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Created dataset: {DATASET}")

def load_to_bigquery(df):
    client   = get_client()
    ensure_dataset_exists(client)

    table_ref = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition = "WRITE_APPEND",
        autodetect = True
    )
    
    job = client.load_table_from_dataframe(df, table_ref, job_config = job_config)
    job.result()

    print(f"Loaded {len(df)} rows into {table_ref}")


