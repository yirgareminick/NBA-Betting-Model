"""
Simple model training placeholder for NBA betting prediction.

This is a basic implementation to complete the pipeline.
In production, this would be replaced with more sophisticated ML models.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import yaml
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class NBAModelTrainer:
    """Simple NBA betting model trainer."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self.feature_columns = None
        self.project_root = Path(__file__).parent.parent.parent
        self.model_dir = self.project_root / "models"
        self.model_dir.mkdir(exist_ok=True)

    def load_features(self) -> pd.DataFrame:
        """Load processed features from parquet file."""
        features_file = self.project_root / "data" / "processed" / "nba_features.parquet"

        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        df = pl.read_parquet(features_file).to_pandas()
        print(f"✓ Features: {len(df):,} records")
        return df

    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for model training."""


        # Select feature columns (exclude metadata and target)
        exclude_cols = ['game_id', 'game_date', 'team_name', 'opponent', 'target_win', 'venue']
        candidate_features = [col for col in df.columns if col not in exclude_cols]

        # Only include numeric columns
        numeric_features = []
        for col in candidate_features:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)

        self.feature_columns = numeric_features

        # Handle the boolean 'is_home' column separately if it exists
        if 'is_home' in df.columns:
            self.feature_columns.append('is_home')

        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = df['target_win'].copy()

        # Convert boolean columns to numeric
        for col in X.select_dtypes(include=['bool']).columns:
            X[col] = X[col].astype(int)

        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))

        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean(numeric_only=True))

        print(f"✓ Prepared: {len(self.feature_columns)} features, {len(X)} samples")

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame = None,
                   use_temporal_split: bool = True) -> dict:
        """Train the model and return metrics with proper temporal validation."""


        if use_temporal_split and df is not None and 'game_date' in df.columns:
            # Temporal split to prevent data leakage
            print("📅 Using temporal train/test split...")
            df_sorted = df.sort_values('game_date').reset_index(drop=True)
            split_idx = int(len(df_sorted) * 0.8)  # 80% for training

            train_mask = np.arange(len(df_sorted)) < split_idx
            test_mask = np.arange(len(df_sorted)) >= split_idx

            X_train, X_test = X.iloc[train_mask], X.iloc[test_mask]
            y_train, y_test = y.iloc[train_mask], y.iloc[test_mask]

            train_dates = df_sorted.iloc[train_mask]['game_date']
            test_dates = df_sorted.iloc[test_mask]['game_date']
            print(f"   Train: {len(X_train):,} games ({train_dates.min().date()} to {train_dates.max().date()})")
            print(f"   Test:  {len(X_test):,} games ({test_dates.min().date()} to {test_dates.max().date()})")
        else:
            # Fallback to random split (with warning)
            print("⚠️  Using random train/test split (may cause data leakage)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)

        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)

        metrics = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'model_type': 'RandomForestClassifier',
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_importance': dict(zip(
                self.feature_columns,
                [float(x) for x in self.model.feature_importances_]
            ))
        }

        print(f"✓ Model trained: {test_score:.3f} accuracy")

        return metrics

    def save_model(self, metrics: dict) -> tuple:
        """Save the trained model and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_file = self.model_dir / f"nba_model_{timestamp}.joblib"
        joblib.dump(self.model, model_file)

        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'model_file': str(model_file),
            'feature_columns': self.feature_columns,
            'metrics': metrics,
            'config': self.config
        }

        metadata_file = self.model_dir / f"nba_model_{timestamp}_metadata.yml"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        # Also save as "latest" for easy loading
        latest_model = self.model_dir / "nba_model_latest.joblib"
        latest_metadata = self.model_dir / "nba_model_latest_metadata.yml"

        joblib.dump(self.model, latest_model)
        with open(latest_metadata, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        print(f"💾 Model saved to: {model_file}")
        print(f"📄 Metadata saved to: {metadata_file}")

        return model_file, metadata_file

    def train_and_save(self, use_temporal_split: bool = True) -> dict:
        """Complete training pipeline."""
        print("=" * 80)
        print("🏀 NBA MODEL TRAINING PIPELINE")
        print("=" * 80)

        # Load data
        df = self.load_features()

        # Prepare data
        X, y = self.prepare_training_data(df)

        # Train model with temporal validation
        metrics = self.train_model(X, y, df, use_temporal_split)

        # Save model
        model_file, metadata_file = self.save_model(metrics)

        print("=" * 80)
        print("✅ MODEL TRAINING COMPLETED")
        print(f"📊 Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"📁 Model File: {model_file}")
        print("=" * 80)

        return metrics


def train_model(config: dict = None) -> dict:
    """Convenience function to train the NBA model."""
    trainer = NBAModelTrainer(config)
    return trainer.train_and_save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA Model Training")
    parser.add_argument("--config", type=Path, help="Path to config YAML file")
    args = parser.parse_args()

    config = {}
    if args.config and args.config.exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    metrics = train_model(config)
    print(f"\n📋 Final metrics:")
    print(f"  - Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"  - CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    print(f"  - Features: {metrics['n_features']}")
