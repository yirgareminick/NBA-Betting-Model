"""
Feature Engineering for NBA Betting Model

This module combines game data, team stats, and odds data to create features
for machine learning models predicting NBA moneyline outcomes.

Data Sources:
- Games: CSV from Kaggle NBA dataset (via ingest_games_new.py)
- Team Stats: Basketball Reference season stats (via ingest_team_stats.py) 
- Odds: The Odds API real-time odds (via ingest_odds.py)
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml


class FeatureEngineer:
    """Feature engineering class for NBA betting model"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.data_paths = self._setup_paths()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML files"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'lookback_games': 10,
            'season_lookback_games': 5,
            'min_games_for_rolling': 3,
            'feature_engineering': {
                'include_team_stats': True,
                'include_odds': True,
                'include_rest_days': True,
                'include_home_advantage': True,
                'include_season_trends': True
            }
        }
    
    def _setup_paths(self) -> Dict[str, Path]:
        """Setup data paths"""
        project_root = Path(__file__).parent.parent.parent
        return {
            'games': project_root / 'data' / 'processed' / 'games_2020_2023.csv',
            'team_stats_dir': project_root / 'data' / 'raw',
            'odds': project_root / 'data' / 'raw' / 'odds_basketball_nba_us.csv',
            'output': project_root / 'data' / 'processed'
        }
    
    def load_games_data(self) -> pl.DataFrame:
        """Load and clean games data from CSV"""
        print("📊 Loading games data...")
        
        if not self.data_paths['games'].exists():
            raise FileNotFoundError(f"Games data not found at {self.data_paths['games']}")
        
        # Load with polars for better performance
        df = pl.read_csv(self.data_paths['games'])
        
        # Clean and standardize
        df = df.with_columns([
            # Parse game date
            pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d").alias("game_date"),
            
            # Create unique game identifier 
            pl.col("game_id").cast(pl.Utf8).alias("game_id"),
            
            # Standardize team names (use strip_chars for polars)
            pl.col("team_name_home").str.strip_chars().alias("team_name_home"),
            pl.col("team_name_away").str.strip_chars().alias("team_name_away"),
            
            # Create win indicators
            (pl.col("pts_home") > pl.col("pts_away")).alias("home_win"),
            (pl.col("pts_away") > pl.col("pts_home")).alias("away_win"),
            
            # Calculate point differential
            (pl.col("pts_home") - pl.col("pts_away")).alias("point_diff")
        ])
        
        print(f"✓ Loaded {len(df):,} games from {df['game_date'].min()} to {df['game_date'].max()}")
        return df
    
    def create_team_game_features(self, games_df: pl.DataFrame) -> pl.DataFrame:
        """Create features by exploding games into team-level records"""
        print("🔧 Creating team-game features...")
        
        # Create home team records
        home_games = games_df.select([
            pl.col("game_id"),
            pl.col("game_date"),
            pl.col("season_id"),
            pl.col("team_name_home").alias("team_name"),
            pl.col("team_abbreviation_home").alias("team_abbrev"),
            pl.lit("home").alias("venue"),
            pl.col("pts_home").alias("pts_for"),
            pl.col("pts_away").alias("pts_against"),
            pl.col("home_win").alias("win"),
            
            # Advanced stats if available
            pl.col("fg_pct_home").alias("fg_pct"),
            pl.col("fg3_pct_home").alias("fg3_pct"),
            pl.col("ft_pct_home").alias("ft_pct"),
            pl.col("reb_home").alias("rebounds"),
            pl.col("ast_home").alias("assists"),
            pl.col("stl_home").alias("steals"),
            pl.col("blk_home").alias("blocks"),
            pl.col("tov_home").alias("turnovers"),
            
            # Opponent info
            pl.col("team_name_away").alias("opponent"),
            pl.col("team_abbreviation_away").alias("opp_abbrev")
        ])
        
        # Create away team records
        away_games = games_df.select([
            pl.col("game_id"),
            pl.col("game_date"),
            pl.col("season_id"),
            pl.col("team_name_away").alias("team_name"),
            pl.col("team_abbreviation_away").alias("team_abbrev"),
            pl.lit("away").alias("venue"),
            pl.col("pts_away").alias("pts_for"),
            pl.col("pts_home").alias("pts_against"),
            pl.col("away_win").alias("win"),
            
            # Advanced stats
            pl.col("fg_pct_away").alias("fg_pct"),
            pl.col("fg3_pct_away").alias("fg3_pct"),
            pl.col("ft_pct_away").alias("ft_pct"),
            pl.col("reb_away").alias("rebounds"),
            pl.col("ast_away").alias("assists"),
            pl.col("stl_away").alias("steals"),
            pl.col("blk_away").alias("blocks"),
            pl.col("tov_away").alias("turnovers"),
            
            # Opponent info
            pl.col("team_name_home").alias("opponent"),
            pl.col("team_abbreviation_home").alias("opp_abbrev")
        ])
        
        # Combine and sort
        team_games = pl.concat([home_games, away_games])
        team_games = team_games.sort(["team_name", "game_date"])
        
        # Calculate derived features
        team_games = team_games.with_columns([
            # Point differential
            (pl.col("pts_for") - pl.col("pts_against")).alias("point_diff"),
            
            # Win as numeric
            pl.col("win").cast(pl.Float32).alias("win_numeric"),
            
            # Home advantage indicator
            (pl.col("venue") == "home").alias("is_home")
        ])
        
        print(f"✓ Created {len(team_games):,} team-game records")
        return team_games
    
    def create_rolling_features(self, team_games: pl.DataFrame, 
                              lookback: int = None) -> pl.DataFrame:
        """Create rolling statistics for each team"""
        print("🔧 Creating rolling features...")
        
        if lookback is None:
            lookback = self.config.get('lookback_games', 10)
        
        # Sort by team and date
        team_games = team_games.sort(["team_name", "game_date"])
        
        # Create rolling features using window functions
        rolling_features = team_games.with_columns([
            # Rolling averages (excluding current game)
            pl.col("pts_for").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_pts_last_{lookback}"),
            pl.col("pts_against").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_pts_allowed_last_{lookback}"),
            pl.col("point_diff").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_point_diff_last_{lookback}"),
            pl.col("win_numeric").shift(1).rolling_mean(lookback).over("team_name").alias(f"win_pct_last_{lookback}"),
            
            # Rolling shooting percentages
            pl.col("fg_pct").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_fg_pct_last_{lookback}"),
            pl.col("fg3_pct").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_fg3_pct_last_{lookback}"),
            pl.col("ft_pct").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_ft_pct_last_{lookback}"),
            
            # Rolling advanced stats
            pl.col("rebounds").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_rebounds_last_{lookback}"),
            pl.col("assists").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_assists_last_{lookback}"),
            pl.col("turnovers").shift(1).rolling_mean(lookback).over("team_name").alias(f"avg_turnovers_last_{lookback}"),
            
            # Recent form (last 5 games)
            pl.col("win_numeric").shift(1).rolling_mean(5).over("team_name").alias("win_pct_last_5"),
            pl.col("point_diff").shift(1).rolling_mean(5).over("team_name").alias("avg_point_diff_last_5"),
            
            # Home/Away splits
            pl.when(pl.col("venue") == "home")
              .then(pl.col("win_numeric").shift(1))
              .otherwise(None)
              .rolling_mean(lookback).over("team_name").alias(f"home_win_pct_last_{lookback}"),
              
            pl.when(pl.col("venue") == "away")
              .then(pl.col("win_numeric").shift(1))
              .otherwise(None)
              .rolling_mean(lookback).over("team_name").alias(f"away_win_pct_last_{lookback}"),
              
            # Games count for validation
            pl.col("game_id").shift(1).count().over("team_name").alias("games_played")
        ])
        
        print(f"✓ Added rolling features with {lookback}-game lookback")
        return rolling_features
    
    def add_rest_days(self, team_games: pl.DataFrame) -> pl.DataFrame:
        """Calculate rest days between games for each team"""
        print("🔧 Adding rest days...")
        
        team_games = team_games.with_columns([
            # Calculate days since last game
            (pl.col("game_date") - pl.col("game_date").shift(1).over("team_name")).dt.total_days().alias("rest_days")
        ])
        
        # Fill first game of season with reasonable default
        team_games = team_games.with_columns([
            pl.col("rest_days").fill_null(2).alias("rest_days")  # Assume 2 days rest for first game
        ])
        
        return team_games
    
    def add_season_trends(self, team_games: pl.DataFrame) -> pl.DataFrame:
        """Add season-long trend features"""
        print("🔧 Adding season trends...")
        
        team_games = team_games.with_columns([
            # Season win percentage (excluding current game)
            pl.col("win_numeric").shift(1).mean().over(["team_name", "season_id"]).alias("season_win_pct"),
            
            # Season averages
            pl.col("pts_for").shift(1).mean().over(["team_name", "season_id"]).alias("season_avg_pts"),
            pl.col("pts_against").shift(1).mean().over(["team_name", "season_id"]).alias("season_avg_pts_allowed"),
            
            # Game number in season
            pl.col("game_date").rank().over(["team_name", "season_id"]).alias("game_number_in_season")
        ])
        
        return team_games
    
    def prepare_final_features(self, team_games: pl.DataFrame) -> pl.DataFrame:
        """Prepare final feature matrix for modeling"""
        print("🔧 Preparing final features...")
        
        # Filter out games without sufficient history
        min_games = self.config.get('min_games_for_rolling', 3)
        filtered = team_games.filter(pl.col("games_played") >= min_games)
        
        # Create target variable (for training)
        filtered = filtered.with_columns([
            pl.col("win").alias("target_win")
        ])
        
        # Select features for modeling
        feature_columns = [
            "game_id", "game_date", "team_name", "opponent", "venue", "is_home",
            "target_win",  
            
            # Rolling performance features
            f"avg_pts_last_{self.config.get('lookback_games', 10)}",
            f"avg_pts_allowed_last_{self.config.get('lookback_games', 10)}",
            f"avg_point_diff_last_{self.config.get('lookback_games', 10)}",
            f"win_pct_last_{self.config.get('lookback_games', 10)}",
            
            # Recent form
            "win_pct_last_5",
            "avg_point_diff_last_5",
            
            # Rest and schedule
            "rest_days",
            "game_number_in_season",
            
            # Season context
            "season_win_pct",
            "season_avg_pts",
            "season_avg_pts_allowed"
        ]
        
        # Only include columns that exist
        available_columns = [col for col in feature_columns if col in filtered.columns]
        features = filtered.select(available_columns)
        
        print(f"✓ Prepared {len(features):,} feature records with {len(available_columns)} features")
        return features
    
    def build_features(self, output_file: str = None) -> pl.DataFrame:
        """Main feature engineering pipeline"""
        print("=" * 80)
        print("🏀 NBA FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
        
        # Load data
        games_df = self.load_games_data()
        
        # Create team-game level features
        team_games = self.create_team_game_features(games_df)
        
        # Add rolling features
        team_games = self.create_rolling_features(team_games)
        
        # Add additional features
        team_games = self.add_rest_days(team_games)
        team_games = self.add_season_trends(team_games)
        
        # Prepare final feature matrix
        features = self.prepare_final_features(team_games)
        
        # Save results
        if output_file is None:
            output_file = self.data_paths['output'] / 'nba_features.parquet'
        
        features.write_parquet(output_file)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'config': self.config,
            'records': len(features),
            'features': features.columns,
            'date_range': {
                'start': str(features['game_date'].min()),
                'end': str(features['game_date'].max())
            }
        }
        
        metadata_file = str(output_file).replace('.parquet', '_metadata.yml')
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print("=" * 80)
        print("✅ FEATURE ENGINEERING COMPLETED")
        print(f"📊 Records: {len(features):,}")
        print(f"🔧 Features: {len(features.columns)}")
        print(f"📁 Output: {output_file}")
        print(f"📄 Metadata: {metadata_file}")
        print("=" * 80)
        
        return features


def build_features(config_path: Optional[Path] = None, 
                  output_file: Optional[str] = None) -> pl.DataFrame:
    """Convenience function to run feature engineering"""
    engineer = FeatureEngineer(config_path)
    return engineer.build_features(output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NBA Feature Engineering Pipeline")
    parser.add_argument("--config", type=Path, help="Path to config YAML file")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--lookback", type=int, default=10, help="Lookback games for rolling features")
    
    args = parser.parse_args()
    
    # Override config with command line args
    if args.config and args.config.exists():
        config_path = args.config
    else:
        config_path = None
    
    features = build_features(config_path, args.output)
    print(f"\n📋 Sample features:")
    print(features.head().to_pandas().to_string())
