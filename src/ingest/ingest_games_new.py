#!/usr/bin/env python3
"""
NBA Basketball Games Data Ingestion Script
Downloads and processes NBA games data from Kaggle with project configuration support
"""

import kagglehub
import pandas as pd
import sqlite3
import sys
import os
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

class NBADataIngestion:
    """NBA Data Ingestion class with configuration management"""

    def __init__(self):
        self.config = self.load_config()
        self.data_paths = self.setup_data_paths()

    def load_config(self) -> Dict:
        """Load configuration from project config files"""
        script_path = Path(__file__)
        project_root = script_path.parent.parent.parent

        config = {'project_root': str(project_root)}

        config_files = {
            'paths': project_root / 'configs' / 'paths.yml',
            'model': project_root / 'configs' / 'model.yml'
        }

        for config_type, config_path in config_files.items():
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config[config_type] = yaml.safe_load(f)
                    print(f"✓ Loaded {config_type} config from {config_path}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load {config_type} config: {e}")
            else:
                print(f"⚠️  Config file not found: {config_path}")

        return config

    def setup_data_paths(self) -> Dict[str, Path]:
        """Setup and create data directory paths"""
        project_root = Path(self.config['project_root'])

        # Get paths from config or use defaults
        if 'paths' in self.config and self.config['paths'] and 'data' in self.config['paths']:
            data_config = self.config['paths']['data']
            paths = {
                'raw': project_root / data_config.get('raw', 'data/raw'),
                'interim': project_root / data_config.get('interim', 'data/interim'),
                'processed': project_root / data_config.get('processed', 'data/processed')
            }
        else:
            # Default structure
            paths = {
                'raw': project_root / 'data' / 'raw',
                'interim': project_root / 'data' / 'interim',
                'processed': project_root / 'data' / 'processed'
            }

        # Create directories
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        return paths

    def download_kaggle_dataset(self) -> str:
        """Download the Kaggle basketball dataset"""
        print("📥 Downloading Kaggle dataset...")
        try:
            dataset_path = kagglehub.dataset_download("wyattowalsh/basketball")
            print(f"✓ Dataset downloaded to: {dataset_path}")
            return dataset_path
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")

    def copy_database_to_project(self, source_path: str) -> Path:
        """Copy the SQLite database to the project's raw data directory"""
        source_db = Path(source_path) / "nba.sqlite"
        target_db = self.data_paths['raw'] / "nba.sqlite"

        if source_db.exists():
            print(f"📁 Copying database to project: {target_db}")
            shutil.copy2(source_db, target_db)
            return target_db
        else:
            raise FileNotFoundError(f"SQLite database not found at {source_db}")

    def get_available_tables(self, db_path: Path) -> List[str]:
        """Get list of available tables in SQLite database"""
        with sqlite3.connect(db_path) as conn:
            tables_df = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
                conn
            )
            return tables_df['name'].tolist()

    def load_data_from_sqlite(self, db_path: Path, table_name: str = None) -> pd.DataFrame:
        """Load data from SQLite database"""
        with sqlite3.connect(db_path) as conn:
            if table_name is None:
                # Find the best table for games data
                tables = self.get_available_tables(db_path)
                games_tables = [t for t in tables if 'game' in t.lower()]

                if games_tables:
                    table_name = games_tables[0]
                    print(f"📊 Using table: {table_name}")
                else:
                    table_name = tables[0] if tables else None
                    print(f"📊 Using first available table: {table_name}")

            if table_name is None:
                raise ValueError("No tables found in database")

            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)

        return df

    def load_data_from_csv(self, source_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        csv_path = Path(source_path) / "game.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"game.csv not found at {csv_path}")

        print(f"📊 Loading data from: {csv_path}")
        return pd.read_csv(csv_path)

    def identify_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify potential date columns in the dataframe"""
        date_keywords = ['date', 'year', 'season', 'game_date']
        date_columns = []

        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                date_columns.append(col)

        return date_columns

    def filter_by_year_range(self, df: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
        """Filter dataframe by year range using various date column strategies"""
        date_columns = self.identify_date_columns(df)

        if not date_columns:
            print("⚠️  No date columns found. Returning full dataset.")
            return df

        print(f"📅 Found date columns: {date_columns}")

        for col in date_columns:
            try:
                print(f"🔍 Attempting to filter using column: {col}")

                if self._is_year_column(df[col]):
                    filtered_df = self._filter_by_year_column(df, col, year_start, year_end)
                elif self._is_date_column(df[col]):
                    filtered_df = self._filter_by_date_column(df, col, year_start, year_end)
                else:
                    continue

                if not filtered_df.empty:
                    print(f"✓ Successfully filtered using {col}: {len(filtered_df)} records")
                    return filtered_df

            except Exception as e:
                print(f"❌ Failed to filter using {col}: {e}")
                continue

        print("⚠️  Could not filter by year range. Returning full dataset.")
        return df

    def _is_year_column(self, series: pd.Series) -> bool:
        """Check if series contains year data"""
        if series.dtype in ['int64', 'int32', 'float64']:
            return True

        # Check for season format like "2020-21"
        sample_str = str(series.iloc[0]) if not series.empty else ""
        return '-' in sample_str and len(sample_str) <= 10

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if series contains date data"""
        if series.empty:
            return False

        # Check if it's already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # Try to parse a few samples
        try:
            sample_size = min(5, len(series))
            parsed = pd.to_datetime(series.iloc[:sample_size], errors='coerce')
            valid_dates = parsed.notna().sum()
            return valid_dates > 0  # At least one valid date
        except Exception:
            return False

    def _filter_by_year_column(self, df: pd.DataFrame, col: str, year_start: int, year_end: int) -> pd.DataFrame:
        """Filter by year column (numeric or season format)"""
        if df[col].dtype in ['int64', 'int32', 'float64']:
            mask = (df[col] >= year_start) & (df[col] <= year_end)
        else:
            # Handle season format like "2020-21"
            df_temp = df.copy()
            df_temp['temp_year'] = df_temp[col].astype(str).str.split('-').str[0]
            df_temp['temp_year'] = pd.to_numeric(df_temp['temp_year'], errors='coerce')
            mask = (df_temp['temp_year'] >= year_start) & (df_temp['temp_year'] <= year_end)

        return df[mask].copy()

    def _filter_by_date_column(self, df: pd.DataFrame, col: str, year_start: int, year_end: int) -> pd.DataFrame:
        """Filter by date column"""
        df_temp = df.copy()
        df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')
        df_temp = df_temp.dropna(subset=[col])
        df_temp['temp_year'] = df_temp[col].dt.year
        mask = (df_temp['temp_year'] >= year_start) & (df_temp['temp_year'] <= year_end)

        return df_temp[mask].drop('temp_year', axis=1).copy()

    def save_data_and_metadata(self, df: pd.DataFrame, year_start: int, year_end: int,
                              data_source: str) -> Tuple[Path, Path]:
        """Save processed data and metadata"""
        # Save data
        output_file = self.data_paths['processed'] / f"games_{year_start}_{year_end}.csv"
        df.to_csv(output_file, index=False)

        # Save metadata
        metadata = {
            'source': 'kaggle:wyattowalsh/basketball',
            'year_range': f"{year_start}-{year_end}",
            'records': len(df),
            'columns': list(df.columns),
            'date_columns': self.identify_date_columns(df),
            'created_at': datetime.now().isoformat(),
            'data_source': data_source,
            'file_size_mb': round(output_file.stat().st_size / (1024*1024), 2)
        }

        metadata_file = self.data_paths['processed'] / f"games_{year_start}_{year_end}_metadata.yml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        return output_file, metadata_file

    def ingest_games_data(self, year_start: int, year_end: int,
                         use_sqlite: bool = True) -> Optional[Tuple[pd.DataFrame, str]]:
        """Main data ingestion method"""
        try:
            # Download dataset
            dataset_path = self.download_kaggle_dataset()

            # Load data
            if use_sqlite:
                db_path = self.copy_database_to_project(dataset_path)
                df = self.load_data_from_sqlite(db_path)
                data_source = "sqlite"
            else:
                df = self.load_data_from_csv(dataset_path)
                data_source = "csv"

            print(f"📊 Original dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

            # Filter by year range
            filtered_df = self.filter_by_year_range(df, year_start, year_end)
            print(f"📊 Filtered dataset: {filtered_df.shape[0]:,} rows")

            if filtered_df.empty:
                print(f"❌ No data found for years {year_start}-{year_end}")
                return None

            # Save results
            output_file, metadata_file = self.save_data_and_metadata(
                filtered_df, year_start, year_end, data_source
            )

            print(f"✅ Data saved to: {output_file}")
            print(f"📄 Metadata saved to: {metadata_file}")

            return filtered_df, str(output_file)

        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            return None

def main():
    """Main execution function"""
    # Parse command line arguments
    year_start = int(sys.argv[1]) if len(sys.argv) > 1 else 2020
    year_end = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
    use_sqlite = sys.argv[3].lower() in ['true', '1', 'yes', 'sqlite'] if len(sys.argv) > 3 else True

    print("=" * 80)
    print("🏀 NBA GAMES DATA INGESTION")
    print("=" * 80)
    print(f"📅 Year range: {year_start} - {year_end}")
    print(f"💾 Data source: {'SQLite database' if use_sqlite else 'CSV files'}")
    print("=" * 80)

    # Validate inputs
    if year_start > year_end:
        print("❌ Error: Start year cannot be greater than end year")
        sys.exit(1)

    # Initialize ingestion
    ingestion = NBADataIngestion()

    # Process data
    result = ingestion.ingest_games_data(year_start, year_end, use_sqlite)

    if result:
        df, output_file = result
        print("=" * 80)
        print("🎉 INGESTION COMPLETED SUCCESSFULLY")
        print(f"📊 Records processed: {len(df):,}")
        print(f"📁 Output file: {output_file}")
        if os.path.exists(output_file):
            print(f"💾 File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        print("=" * 80)

        # Show sample
        print("\n📋 Sample of processed data:")
        print(df.head().to_string())

        # Show basic stats
        print(f"\n📈 Data Summary:")
        print(f"   • Columns: {df.shape[1]}")
        print(f"   • Date range: {year_start}-{year_end}")
        print(f"   • Missing values: {df.isnull().sum().sum():,}")

    else:
        print("=" * 80)
        print("❌ INGESTION FAILED")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()