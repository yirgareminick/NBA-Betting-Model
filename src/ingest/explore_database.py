#!/usr/bin/env python3
"""
Database Explorer Script
Explores the Kaggle basketball dataset to understand its structure
"""

import kagglehub
import pandas as pd
import sqlite3
import os

def explore_dataset():
    """
    Explore the Kaggle basketball dataset to understand its structure
    """
    
    print("="*60)
    print("EXPLORING KAGGLE BASKETBALL DATASET")
    print("="*60)
    
    try:
        # First, let's see what files are available in the dataset
        print("Attempting to download dataset...")
        
        # Try to download the dataset without specifying a file
        # This should give us the full dataset structure
        dataset_path = kagglehub.dataset_download("wyattowalsh/basketball")
        print(f"Dataset downloaded to: {dataset_path}")
        
        # List all files in the dataset directory
        print("\nFiles in dataset:")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size / (1024*1024):.2f} MB)")
        
        # Look for SQLite database files
        db_files = [f for f in files if f.endswith(('.db', '.sqlite', '.sqlite3'))]
        
        if db_files:
            print(f"\nFound SQLite databases: {db_files}")
            
            # Explore the first database file
            db_file = db_files[0]
            db_path = os.path.join(dataset_path, db_file)
            
            print(f"\nExploring database: {db_file}")
            conn = sqlite3.connect(db_path)
            
            # Get all tables
            tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(tables_query, conn)
            
            print(f"\nTables in database ({len(tables)} total):")
            for table in tables['name']:
                print(f"  - {table}")
            
            # For each table, show structure and sample data
            for table_name in tables['name'][:5]:  # Limit to first 5 tables
                print(f"\n" + "="*40)
                print(f"TABLE: {table_name}")
                print("="*40)
                
                try:
                    # Get table schema
                    schema_query = f"PRAGMA table_info({table_name});"
                    schema = pd.read_sql_query(schema_query, conn)
                    
                    print("Columns:")
                    for _, col in schema.iterrows():
                        print(f"  - {col['name']} ({col['type']})")
                    
                    # Get row count
                    count_query = f"SELECT COUNT(*) as count FROM {table_name};"
                    count = pd.read_sql_query(count_query, conn)
                    print(f"\nRow count: {count['count'].iloc[0]:,}")
                    
                    # Show sample data
                    sample_query = f"SELECT * FROM {table_name} LIMIT 3;"
                    sample = pd.read_sql_query(sample_query, conn)
                    print(f"\nSample data:")
                    print(sample.to_string(index=False))
                    
                except Exception as e:
                    print(f"Error exploring table {table_name}: {str(e)}")
            
            conn.close()
            
        else:
            print("\nNo SQLite database files found.")
            print("Available files:")
            for file in files:
                print(f"  - {file}")
    
    except Exception as e:
        print(f"Error exploring dataset: {str(e)}")
        
        # Try alternative approach - use the deprecated load_dataset method
        print("\nTrying alternative approach...")
        try:
            # Try to load without specifying a file path
            result = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "wyattowalsh/basketball"
            )
            print(f"Loaded data type: {type(result)}")
            if hasattr(result, 'shape'):
                print(f"Data shape: {result.shape}")
                print(f"Columns: {list(result.columns)}")
                print("\nFirst few rows:")
                print(result.head())
        except Exception as e2:
            print(f"Alternative approach also failed: {str(e2)}")

if __name__ == "__main__":
    explore_dataset()