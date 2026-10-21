"""
Simple model training placeholder for NBA betting prediction.

This is a basic implementation to complete the pipeline.
In production, this would be replaced with more sophisticated ML models.
"""

import hashlib
import logging
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import yaml
import warnings

logger = logging.getLogger(__name__)
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
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_features(self) -> pd.DataFrame:
        """Load processed features from parquet file."""
        features_file = self.project_root / "data" / "processed" / "nba_features.parquet"

        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        df = pl.read_parquet(features_file).to_pandas()
        logger.info("✓ Features: %s records", len(df))
        return df

    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for model training."""


        # Select feature columns (exclude metadata and target)
        exclude_cols = ['game_id', 'game_date', 'team_name', 'opponent', 'target_win', 'venue']
        candidate_features = [col for col in df.columns if col not in exclude_cols]

        # Only include numeric columns while preserving order and removing duplicates
        numeric_features = []
        for col in candidate_features:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)

        self.feature_columns = list(dict.fromkeys(numeric_features))

        # Handle the boolean 'is_home' column separately if it exists, without duplicating it
        if 'is_home' in df.columns and 'is_home' not in self.feature_columns:
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

        logger.info("✓ Prepared: %s features, %s samples", len(self.feature_columns), len(X))

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame = None,
                   use_temporal_split: bool = True) -> dict:
        """Train the best-performing model and return metrics with proper temporal validation."""

        X = X.copy()
        y = y.copy()

        if df is not None and 'game_date' in df.columns:
            df_sorted = df.sort_values('game_date').reset_index(drop=True)
            sorted_index = df_sorted.index
            X = X.loc[sorted_index].copy()
            y = y.loc[sorted_index].copy()

        if use_temporal_split and df is not None and 'game_date' in df.columns:
            logger.info("📅 Using temporal train/test split...")
            df_sorted = df.sort_values('game_date').reset_index(drop=True)
            split_idx = max(1, int(len(df_sorted) * 0.8))
            split_idx = min(split_idx, len(df_sorted) - 1)

            X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
            y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

            train_dates = df_sorted.iloc[:split_idx]['game_date']
            test_dates = df_sorted.iloc[split_idx:]['game_date']
            logger.info("   Train: %s games (%s to %s)", len(X_train), train_dates.min().date(), train_dates.max().date())
            logger.info("   Test:  %s games (%s to %s)", len(X_test), test_dates.min().date(), test_dates.max().date())
        else:
            logger.warning("⚠️  Using random train/test split (may cause data leakage)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        candidate_models = {
            'logistic_regression': LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                solver='lbfgs'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
        }

        def build_pipeline(model):
            return Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', model),
            ])

        best_name = None
        best_pipeline = None
        best_cv_mean = -np.inf

        for name, model in candidate_models.items():
            pipeline = build_pipeline(model)
            cv = TimeSeriesSplit(n_splits=min(3, max(2, len(X_train) // 10))) if use_temporal_split and df is not None and 'game_date' in df.columns else 3
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
            mean_score = float(cv_scores.mean())
            logger.info("📊 %s CV accuracy: %.4f", name, mean_score)
            if mean_score > best_cv_mean:
                best_cv_mean = mean_score
                best_name = name
                best_pipeline = pipeline

        if best_pipeline is None:
            raise ValueError("No candidate models were available for training")

        self.model = best_pipeline
        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        cv = TimeSeriesSplit(n_splits=min(3, max(2, len(X_train) // 10))) if use_temporal_split and df is not None and 'game_date' in df.columns else 3
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv)

        metrics = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'model_type': best_name,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
        }

        model_step = self.model.named_steps['model'] if hasattr(self.model, 'named_steps') else self.model
        if hasattr(model_step, 'feature_importances_'):
            metrics['feature_importance'] = dict(zip(
                self.feature_columns,
                [float(x) for x in model_step.feature_importances_]
            ))

        logger.info("✓ Selected model: %s with test accuracy %.3f", best_name, test_score)

        return metrics

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Compute a SHA256 checksum for a file."""
        digest = hashlib.sha256()
        with open(path, 'rb') as file_handle:
            for chunk in iter(lambda: file_handle.read(8192), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def save_model(self, metrics: dict) -> tuple:
        """Save the trained model and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = getattr(self, 'config', {}) or {}
        project_root = getattr(self, 'project_root', Path(__file__).resolve().parents[2])
        model_dir = getattr(self, 'model_dir', project_root / 'models')
        feature_columns = getattr(self, 'feature_columns', []) or []

        # Save model
        model_file = model_dir / f"nba_model_{timestamp}.joblib"
        joblib.dump(self.model, model_file)
        model_sha256 = self._file_sha256(model_file)

        # Save metadata
        model_type = self.model.__class__.__name__ if self.model is not None else 'unknown'
        metadata = {
            'created_at': datetime.now().isoformat(),
            'model_file': str(model_file.name),
            'model_type': model_type,
            'model_sha256': model_sha256,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'config': config
        }

        metadata_file = model_dir / f"nba_model_{timestamp}_metadata.yml"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        # Also save as "latest" for easy loading
        latest_model = model_dir / "nba_model_latest.joblib"
        latest_metadata = model_dir / "nba_model_latest_metadata.yml"

        joblib.dump(self.model, latest_model)
        latest_metadata_payload = dict(metadata)
        latest_metadata_payload['model_file'] = str(latest_model.name)
        latest_metadata_payload['model_type'] = model_type
        latest_metadata_payload['model_sha256'] = self._file_sha256(latest_model)
        with open(latest_metadata, 'w', encoding='utf-8') as f:
            yaml.dump(latest_metadata_payload, f, default_flow_style=False)

        logger.info("💾 Model saved to: %s", model_file)
        logger.info("📄 Metadata saved to: %s", metadata_file)

        return model_file, metadata_file

    def train_and_save(self, use_temporal_split: bool = True) -> dict:
        """Complete training pipeline."""
        logger.info("%s", "=" * 80)
        logger.info("🏀 NBA MODEL TRAINING PIPELINE")
        logger.info("%s", "=" * 80)

        # Load data
        df = self.load_features()

        # Prepare data
        X, y = self.prepare_training_data(df)

        # Train model with temporal validation
        metrics = self.train_model(X, y, df, use_temporal_split)

        # Save model
        model_file, metadata_file = self.save_model(metrics)

        logger.info("%s", "=" * 80)
        logger.info("✅ MODEL TRAINING COMPLETED")
        logger.info("📊 Test Accuracy: %s", f"{metrics['test_accuracy']:.3f}")
        logger.info("📁 Model File: %s", model_file)
        logger.info("%s", "=" * 80)

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
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    metrics = train_model(config)
    logger.info("\n📋 Final metrics:")
    logger.info("  - Accuracy: %s", f"{metrics['test_accuracy']:.3f}")
    logger.info("  - CV Score: %s ± %s", f"{metrics['cv_mean']:.3f}", f"{metrics['cv_std']:.3f}")
    logger.info("  - Features: %s", metrics['n_features'])
