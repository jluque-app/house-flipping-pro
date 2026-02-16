
import pandas as pd
from sqlalchemy import create_engine, text
import os
import sys

# Database connection
DB_USER = "jluque" # Adjust as needed
DB_HOST = "localhost"
DB_NAME = "bcn" # Using bcn database as confirmed
DB_PORT = "5432"

DATABASE_URL = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# File path (Local extracted file)
VALENCIA_FILE = "/Users/jluque/.gemini/antigravity/scratch/house_flipping/backend/data/data_ROI_Valencia.xlsx"

def import_data():
    print(f"Reading {VALENCIA_FILE}...")
    try:
        # Check if file exists
        if not os.path.exists(VALENCIA_FILE):
             print(f"Error: File not found at {VALENCIA_FILE}")
             return

        df = pd.read_excel(VALENCIA_FILE, engine='openpyxl')
        print(f"Loaded {len(df)} rows.")
        print(f"Columns: {list(df.columns)}")
        
        # Deduplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Add city column
        df['city'] = 'Valencia'
        
        # Drop existing postal_code if we are going to rename postal_codenum to it
        if 'postal_code' in df.columns and 'postal_codenum' in df.columns:
            df = df.drop(columns=['postal_code'])
        
        # Columns mapping based on previous knowledge and schema
        # We need to map dataframe columns to database columns
        # Database columns: (from \d properties)
        # title, district, neighborhood, postal_code, latitude, longitude, property_type, subtype, size, bedrooms, bathrooms, floor, status, new_construction, price, price_m2, lift, garage, storage, terrace, air_conditioning, swimming_pool, garden, sports, ingreso, vi, vo, comprable, roi, ck, cm
        
        # Excel columns (inferred from data_ROI.xlsx headers):
        # 'title', 'district', 'neighborhood', 'postal_codenum' (postal_code), 'latitude', 'longitude', 'price', 'price_m2', 'size', 'bedrooms', 'bathrooms', 'floor', 'lift', 'garage', 'storage', 'terrace', 'air_conditioning', 'swimming_pool', 'garden', 'sports', 'status', 'new_construction', 'VI', 'VO', 'comprable', 'ROI', 'cm', 'ck'
        
        # Rename columns to match DB
        rename_map = {
            'postal_codenum': 'postal_code',
            'VI': 'vi',
            'VO': 'vo',
            'ROI': 'roi',
            # 'Ingreso Medio por Persona': 'ingreso' -- Check if this exists in Valencia file
        }
        
        # Normalize columns (lowercase)
        df.columns = [c.strip() for c in df.columns]
        
        # Apply renaming
        df = df.rename(columns=rename_map)
        
        # Ensure postal_code is string
        if 'postal_code' in df.columns:
            df['postal_code'] = df['postal_code'].astype(str).str.replace(r'\.0$', '', regex=True)

        # Create geometry
        # format: POINT(lon lat)
        # We'll update this via SQL after insertion or construct it here
        # Easiest: Insert latitude/longitude and let a trigger handle it, or update explicitly.
        # But we manipulate the DB directly via pandas usually without geom.
        # We will insert ignoring 'geom', 'id', 'created_at', 'updated_at'
        
        # Select applicable columns
        db_cols = [
            'title', 'district', 'neighborhood', 'postal_code', 'latitude', 'longitude', 
            'property_type', 'subtype', 'size', 'bedrooms', 'bathrooms', 'floor', 
            'status', 'new_construction', 'price', 'price_m2', 
            'lift', 'garage', 'storage', 'terrace', 'air_conditioning', 'swimming_pool', 'garden', 'sports', 
            'ingreso', 'vi', 'vo', 'comprable', 'roi', 'ck', 'cm', 'city'
        ]
        
        # Filter df to only cols that exist
        cols_to_insert = [c for c in db_cols if c in df.columns]
        
        print(f"Inserting columns: {cols_to_insert}")
        
        final_df = df[cols_to_insert].copy()
        
        # Write to DB
        final_df.to_sql('properties', engine, if_exists='append', index=False, method='multi', chunksize=1000)
        print("Data inserted successfully.")
        
        # Update geometry
        with engine.connect() as conn:
            print("Updating geometry...")
            conn.execute(text("UPDATE properties SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) WHERE city = 'Valencia' AND geom IS NULL;"))
            conn.commit()
            print("Geometry updated.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import_data()
