import pandas as pd
import os
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry, WKTElement
import numpy as np

# Database connection
# Adjust if running outside docker (localhost) or inside (db)
# We assume this script runs inside the container or with proper env vars
DB_USER = os.environ.get("POSTGRES_USER", "bcn")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "bcn")
DB_HOST = "localhost" # Default to localhost for running from host, change if in docker
DB_PORT = "5432"
DB_NAME = os.environ.get("POSTGRES_DB", "bcn")

# If running inside docker compose, the host might be 'db'
if os.environ.get("UseDockerHost"):
    DB_HOST = "db"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

EXCEL_FILE = "../data_ROI_BCN.xlsx" # Path relative to script execution location (scripts/) 
# We might need to adjust this depending on where we put the file. Reviewing task.

def import_data():
    print("Reading Excel file...")
    # Attempt to locate file
    file_path = "backend/data_ROI.xlsx"
    if not os.path.exists(file_path):
        file_path = "../data_ROI.xlsx"
    if not os.path.exists(file_path):
        file_path = "data_ROI.xlsx"
    
    if not os.path.exists(file_path):
        print(f"Error: data_ROI.xlsx not found at {file_path}")
        return

    df = pd.read_excel(file_path)
    
    # Rename columns to match DB schema if necessary
    # Based on PDF mapping:
    # neighborhood -> neighborhood
    # district -> district
    # postal_code -> postal_code
    # price -> price
    # price_m2 -> price_m2
    # size -> size
    # bedrooms -> bedrooms
    # bathrooms -> bathrooms
    # floor -> floor
    # ...
    # prob_propietario_1 -> ck
    # prob_investor_1 -> cm
    # VI -> vi
    # VO -> vo
    # ROI -> roi
    
    print("Transforming data...")
    
    # Filter rows with essential data
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Map columns
    column_mapping = {
        'prob_propietario_1': 'ck',
        'prob_investor_1': 'cm',
        'VI': 'vi',
        'VO': 'vo',
        'ROI': 'roi',
        # Add other mappings if names differ, assuming most match lowersnake case or close
    }
    df = df.rename(columns=column_mapping)
    
    # Create Geometry
    # SRID 4326
    df['geom'] = df.apply(lambda row: WKTElement(f'POINT({row.longitude} {row.latitude})', srid=4326), axis=1)
    
    # Select columns that exist in our table
    valid_columns = [
        'id', 'title', 'district', 'neighborhood', 'postal_code', 
        'latitude', 'longitude', 'geom',
        'property_type', 'subtype', 'size', 'bedrooms', 'bathrooms', 'floor', 'status', 'new_construction',
        'price', 'price_m2',
        'lift', 'garage', 'storage', 'terrace', 'air_conditioning', 'swimming_pool', 'garden', 'sports',
        'ingreso', 'vi', 'vo', 'comprable', 'roi', 'ck', 'cm'
    ]
    
    # Ensure ID is present, if not create one or use index
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
    
    # Fill NaNs where appropriate
    df['comprable'] = df['comprable'].fillna(0)
    
    # Filter columns
    df_final = df[[c for c in valid_columns if c in df.columns]]
    
    print(f"Inserting {len(df_final)} rows...")
    
    # Insert to DB
    df_final.to_sql('properties', engine, if_exists='append', index=False, dtype={'geom': Geometry('POINT', srid=4326)})
    
    print("Data import complete.")

if __name__ == "__main__":
    import_data()
