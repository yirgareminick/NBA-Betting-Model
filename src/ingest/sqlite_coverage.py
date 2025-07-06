#!/usr/bin/env python3
"""
Quick script to check the year coverage of the NBA SQLite database
"""

import sqlite3
import pandas as pd
from pathlib import Path

def check_database_coverage(db_path):
    """Check the year coverage in the NBA database"""
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at: {db_path}")
        print("Make sure you've downloaded and copied the database to your project")
        return
    
    print(f"🔍 Analyzing database: {db_path}")
    print("=" * 60)
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Get all table names
            tables_df = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", 
                conn
            )
            tables = tables_df['name'].tolist()
            print(f"📊 Found {len(tables)} tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}")
            
            # Find tables that likely contain game data with dates
            game_tables = [t for t in tables if any(keyword in t.lower() 
                          for keyword in ['game', 'schedule', 'match'])]
            
            if not game_tables:
                game_tables = tables[:3]  # Check first few tables
            
            print(f"🏀 Checking tables for date ranges: {game_tables}")
            print("-" * 60)
            
            for table in game_tables:
                try:
                    # Get table info
                    columns_df = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
                    columns = columns_df['name'].tolist()
                    
                    # Find date-related columns
                    date_columns = [col for col in columns if any(keyword in col.lower() 
                                   for keyword in ['date', 'year', 'season'])]
                    
                    if not date_columns:
                        continue
                    
                    print(f"\n📅 Table: {table}")
                    print(f"   Columns: {len(columns)} total")
                    print(f"   Date columns: {date_columns}")
                    
                    # Check row count
                    count_df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table};", conn)
                    total_rows = count_df['count'].iloc[0]
                    print(f"   Total rows: {total_rows:,}")
                    
                    # Check date ranges for each date column
                    for date_col in date_columns[:2]:  # Limit to first 2 date columns
                        try:
                            # Get min/max values
                            query = f"""
                            SELECT 
                                MIN({date_col}) as min_date,
                                MAX({date_col}) as max_date,
                                COUNT(DISTINCT {date_col}) as unique_dates
                            FROM {table} 
                            WHERE {date_col} IS NOT NULL
                            """
                            
                            date_range_df = pd.read_sql_query(query, conn)
                            
                            if not date_range_df.empty:
                                min_date = date_range_df['min_date'].iloc[0]
                                max_date = date_range_df['max_date'].iloc[0]
                                unique_dates = date_range_df['unique_dates'].iloc[0]
                                
                                print(f"   📆 {date_col}:")
                                print(f"      Range: {min_date} → {max_date}")
                                print(f"      Unique values: {unique_dates:,}")
                                
                                # Try to extract years if it looks like a date
                                try:
                                    if len(str(min_date)) > 4:  # Likely a full date
                                        sample_df = pd.read_sql_query(
                                            f"SELECT {date_col} FROM {table} WHERE {date_col} IS NOT NULL LIMIT 5", 
                                            conn
                                        )
                                        sample_df[date_col] = pd.to_datetime(sample_df[date_col], errors='coerce')
                                        if not sample_df[date_col].isna().all():
                                            years = sample_df[date_col].dt.year.unique()
                                            print(f"      Sample years: {sorted(years)}")
                                except:
                                    pass
                                    
                        except Exception as e:
                            print(f"   ❌ Error checking {date_col}: {e}")
                    
                except Exception as e:
                    print(f"❌ Error analyzing table {table}: {e}")
            
            # Quick check for specific years in game table if it exists
            if 'game' in tables:
                print(f"\n🎯 Quick year check on 'game' table:")
                try:
                    sample_df = pd.read_sql_query("SELECT * FROM game LIMIT 10", conn)
                    print(f"   Sample columns: {list(sample_df.columns)}")
                    
                    # Look for season or year data
                    for col in sample_df.columns:
                        if 'season' in col.lower() or 'year' in col.lower():
                            unique_vals = sample_df[col].unique()[:10]
                            print(f"   {col} sample values: {unique_vals}")
                            
                except Exception as e:
                    print(f"   ❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Database error: {e}")

def main():
    """Check database coverage"""
    
    # Try multiple possible locations
    possible_paths = [
        "data/raw/nba.sqlite",
        "/Users/yirgareminick/.cache/kagglehub/datasets/wyattowalsh/basketball/versions/231/nba.sqlite",
        "nba.sqlite"
    ]
    
    db_path = None
    for path in possible_paths:
        if Path(path).exists():
            db_path = path
            break
    
    if db_path:
        check_database_coverage(db_path)
    else:
        print("❌ NBA database not found in expected locations:")
        for path in possible_paths:
            print(f"   • {path}")
        print("\nTo check the database:")
        print("1. Make sure you've run the ingestion script")
        print("2. Or specify the path: python check_years.py /path/to/nba.sqlite")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        check_database_coverage(sys.argv[1])
    else:
        main()
        