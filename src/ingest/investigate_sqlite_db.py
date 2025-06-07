#!/usr/bin/env python3
"""
Check the latest year available in the NBA basketball dataset
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from datetime import datetime

def check_latest_year_sqlite():
    """Check latest year using SQLite database"""
    print("🔍 Checking latest year in NBA dataset (SQLite)...")
    print("=" * 60)
    
    try:
        # Load from SQLite database with a query to get recent data
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "wyattowalsh/basketball",
            "nba.sqlite",
            sql_query="""
            SELECT * FROM game 
            ORDER BY game_date DESC 
            LIMIT 100
            """
        )
        
        print(f"✅ Loaded {len(df)} recent records")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Find date columns
        date_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['date', 'year', 'season'])]
        
        print(f"📅 Date columns found: {date_columns}")
        
        # Check each date column for latest year
        for col in date_columns:
            try:
                if 'date' in col.lower():
                    # Handle actual dates
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    latest_date = df[col].max()
                    earliest_date = df[col].min()
                    
                    print(f"\n📆 {col}:")
                    print(f"   Latest: {latest_date}")
                    print(f"   Earliest: {earliest_date}")
                    
                    if pd.notna(latest_date):
                        latest_year = latest_date.year
                        print(f"   Latest Year: {latest_year}")
                
                elif 'year' in col.lower() or 'season' in col.lower():
                    # Handle year/season columns
                    unique_values = df[col].dropna().unique()
                    
                    print(f"\n📅 {col}:")
                    print(f"   Latest values: {sorted(unique_values)[-10:]}")
                    
                    # Try to extract numeric years
                    years = []
                    for val in unique_values:
                        try:
                            if isinstance(val, (int, float)):
                                years.append(int(val))
                            elif isinstance(val, str):
                                # Handle formats like "2023-24"
                                if '-' in val:
                                    year = int(val.split('-')[0])
                                    years.append(year)
                                else:
                                    year = int(val)
                                    years.append(year)
                        except:
                            continue
                    
                    if years:
                        latest_year = max(years)
                        earliest_year = min(years)
                        print(f"   Year range: {earliest_year} - {latest_year}")
                        
            except Exception as e:
                print(f"❌ Error processing {col}: {e}")
    
    except Exception as e:
        print(f"❌ Error loading SQLite data: {e}")
        return None

def check_latest_year_csv():
    """Check latest year using CSV files"""
    print("\n🔍 Checking latest year in NBA dataset (CSV)...")
    print("=" * 60)
    
    try:
        # Load game.csv
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "wyattowalsh/basketball",
            "game.csv"
        )
        
        print(f"✅ Loaded {len(df):,} total game records")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Find date columns
        date_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['date', 'year', 'season'])]
        
        print(f"📅 Date columns found: {date_columns}")
        
        # Get latest data by sorting
        for col in date_columns[:2]:  # Check first 2 date columns
            try:
                print(f"\n📆 Analyzing {col}:")
                
                # Get basic stats
                non_null_count = df[col].count()
                print(f"   Non-null values: {non_null_count:,}")
                
                if 'date' in col.lower():
                    # Handle date columns
                    df_temp = df.copy()
                    df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')
                    df_temp = df_temp.dropna(subset=[col])
                    
                    if not df_temp.empty:
                        latest_date = df_temp[col].max()
                        earliest_date = df_temp[col].min()
                        
                        print(f"   Date range: {earliest_date.date()} → {latest_date.date()}")
                        print(f"   Latest year: {latest_date.year}")
                        
                        # Show some recent games
                        recent = df_temp.nlargest(5, col)
                        print(f"   Recent games:")
                        for _, row in recent.iterrows():
                            print(f"     {row[col].date()}")
                
                else:
                    # Handle year/season columns
                    unique_vals = df[col].dropna().unique()
                    
                    # Sort to find latest
                    try:
                        numeric_vals = []
                        for val in unique_vals:
                            if isinstance(val, (int, float)):
                                numeric_vals.append(val)
                            elif isinstance(val, str) and '-' in val:
                                # Handle "2023-24" format
                                year = int(val.split('-')[0])
                                numeric_vals.append(year)
                        
                        if numeric_vals:
                            latest_year = max(numeric_vals)
                            earliest_year = min(numeric_vals)
                            print(f"   Year range: {earliest_year} - {latest_year}")
                            
                            # Show recent values
                            recent_vals = sorted(set(numeric_vals))[-10:]
                            print(f"   Recent years: {recent_vals}")
                        
                    except Exception as e:
                        print(f"   Sample values: {list(unique_vals)[:10]}")
                        
            except Exception as e:
                print(f"❌ Error processing {col}: {e}")
    
    except Exception as e:
        print(f"❌ Error loading CSV data: {e}")

def check_dataset_info():
    """Get general dataset information"""
    print("\n📋 Dataset Information:")
    print("=" * 60)
    
    try:
        # Download dataset to get file list
        dataset_path = kagglehub.dataset_download("wyattowalsh/basketball")
        print(f"📁 Dataset location: {dataset_path}")
        
        # List files and sizes
        import os
        from pathlib import Path
        
        dataset_dir = Path(dataset_path)
        files = list(dataset_dir.glob("*"))
        
        print(f"\n📄 Available files:")
        for file in sorted(files):
            if file.is_file():
                size_mb = file.stat().st_size / (1024*1024)
                print(f"   {file.name:<25} ({size_mb:.1f} MB)")
        
        print(f"\n🕒 Current date: {datetime.now().strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"❌ Error getting dataset info: {e}")

def main():
    """Main function to check latest year"""
    print("🏀 NBA DATASET YEAR COVERAGE CHECK")
    print("=" * 80)
    
    # Check dataset info first
    check_dataset_info()
    
    # Try SQLite first (more efficient for large datasets)
    try:
        check_latest_year_sqlite()
        print("\n" + "="*80)
        print("✅ SQLite check completed")
    except Exception as e:
        print(f"⚠️  SQLite check failed: {e}")
        print("Falling back to CSV...")
        
        # Fallback to CSV
        try:
            check_latest_year_csv()
            print("\n" + "="*80)
            print("✅ CSV check completed")
        except Exception as e:
            print(f"❌ Both SQLite and CSV checks failed: {e}")

if __name__ == "__main__":
    main()