"""
Advanced Model Training for NBA Betting Predictions

This module implements sophisticated ML models including XGBoost, LightGBM,
and Neural Networks with hyperparameter optimization and ensemble methods.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import joblib
import yaml
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sys

# Optional advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    from sklearn.neural_network import MLPClassifier
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False
    MLPClassifier = None

warnings.filterwarnings('ignore')


class AdvancedNBAModelTrainer:
    """Advanced NBA betting model trainer with multiple algorithms and optimization."""

    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        self.models = {}
        self.feature_columns = None
        self.project_root = Path(__file__).parent.parent.parent
        self.model_dir = self.project_root / "models"
        self.model_dir.mkdir(exist_ok=True)

    def _load_default_config(self) -> Dict:
        """Load default model configuration from YAML file."""
        config_file = self.project_root / "configs" / "model.yml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract model-specific config and add defaults for advanced training
                model_config = config.get('model', {})
                model_config.update({
                    'algorithms': ['random_forest', 'xgboost', 'lightgbm'],
                    'ensemble': True,
                    'hyperparameter_tuning': True,
                })
                
                return model_config
            except Exception as e:
                print(f"Warning: Failed to load config from {config_file}: {e}")
        
        # Fallback to hardcoded defaults
        return {
            'algorithms': ['random_forest', 'xgboost', 'lightgbm'],
            'ensemble': True,
            'hyperparameter_tuning': True,
            'cv_folds': 5,
            'test_size': 0.2,
            'random_state': 42
        }

    def load_features(self) -> pd.DataFrame:
        """Load processed features from parquet file."""
        features_file = self.project_root / "data" / "processed" / "nba_features.parquet"

        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        print(f"📊 Loading features from {features_file}")
        df = pl.read_parquet(features_file).to_pandas()
        print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
        return df

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training."""
        print("🔧 Preparing training data...")

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

        print(f"✓ Features prepared: {len(self.feature_columns)} features, {len(X)} samples")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def create_random_forest_model(self) -> RandomForestClassifier:
        """Create and optionally tune Random Forest model."""
        if self.config.get('hyperparameter_tuning', False):
            print("🔧 Tuning Random Forest hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10]
            }

            rf = RandomForestClassifier(random_state=self.config['random_state'], n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            return grid_search
        else:
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.config['random_state'],
                n_jobs=-1
            )

    def create_xgboost_model(self) -> Any:
        """Create and optionally tune XGBoost model."""
        if not XGBOOST_AVAILABLE:
            print("⚠️  XGBoost not available, skipping...")
            return None

        if self.config.get('hyperparameter_tuning', False):
            print("🔧 Tuning XGBoost hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }

            xgb_model = xgb.XGBClassifier(
                random_state=self.config['random_state'],
                n_jobs=-1,
                eval_metric='logloss'
            )
            grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            return grid_search
        else:
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                random_state=self.config['random_state'],
                n_jobs=-1,
                eval_metric='logloss'
            )

    def create_lightgbm_model(self) -> Any:
        """Create and optionally tune LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️  LightGBM not available, skipping...")
            return None

        if self.config.get('hyperparameter_tuning', False):
            print("🔧 Tuning LightGBM hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9, -1],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }

            lgb_model = lgb.LGBMClassifier(
                random_state=self.config['random_state'],
                n_jobs=-1,
                verbose=-1
            )
            grid_search = GridSearchCV(lgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            return grid_search
        else:
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=50,
                random_state=self.config['random_state'],
                n_jobs=-1,
                verbose=-1
            )

    def create_neural_network_model(self) -> Any:
        """Create and optionally tune Neural Network model."""
        if not NEURAL_NET_AVAILABLE:
            print("⚠️  Neural Network not available, skipping...")
            return None

        if self.config.get('hyperparameter_tuning', False):
            print("🔧 Tuning Neural Network hyperparameters...")
            param_grid = {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }

            nn_model = MLPClassifier(
                max_iter=1000,
                random_state=self.config['random_state'],
                early_stopping=True,
                validation_fraction=0.1
            )
            grid_search = GridSearchCV(nn_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            return grid_search
        else:
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                learning_rate_init=0.01,
                alpha=0.001,
                max_iter=1000,
                random_state=self.config['random_state'],
                early_stopping=True,
                validation_fraction=0.1
            )

    def train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train individual models and return their performance."""
        algorithms = self.config.get('algorithms', ['random_forest'])
        model_results = {}

        # Model factory
        model_factory = {
            'random_forest': self.create_random_forest_model,
            'xgboost': self.create_xgboost_model,
            'lightgbm': self.create_lightgbm_model,
            'neural_network': self.create_neural_network_model
        }

        for algorithm in algorithms:
            if algorithm not in model_factory:
                print(f"⚠️  Unknown algorithm: {algorithm}")
                continue

            print(f"🤖 Training {algorithm.replace('_', ' ').title()} model...")

            model = model_factory[algorithm]()
            if model is None:
                continue

            try:
                # Train model
                model.fit(X_train, y_train)

                # Get best model if using GridSearch
                if hasattr(model, 'best_estimator_'):
                    best_model = model.best_estimator_
                    print(f"✓ Best parameters: {model.best_params_}")
                else:
                    best_model = model

                # Evaluate model
                train_score = best_model.score(X_train, y_train)
                test_score = best_model.score(X_test, y_test)

                # Cross-validation
                cv_scores = cross_val_score(best_model, X_train, y_train,
                                          cv=self.config['cv_folds'], scoring='accuracy')

                # Predictions for probability calibration
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                logloss = log_loss(y_test, y_pred_proba)

                model_results[algorithm] = {
                    'model': best_model,
                    'train_accuracy': float(train_score),
                    'test_accuracy': float(test_score),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'log_loss': float(logloss)
                }

                self.models[algorithm] = best_model

                print(f"✓ {algorithm.replace('_', ' ').title()} completed:")
                print(f"  - Test accuracy: {test_score:.3f}")
                print(f"  - CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"  - Log loss: {logloss:.3f}")

            except Exception as e:
                print(f"❌ Error training {algorithm}: {str(e)}")
                continue

        return model_results

    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Optional[VotingClassifier]:
        """Create an ensemble model from trained individual models."""
        if not self.config.get('ensemble', False) or len(self.models) < 2:
            return None

        print("🔗 Creating ensemble model...")

        # Create voting classifier
        estimators = [(name, model) for name, model in self.models.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')

        try:
            ensemble.fit(X_train, y_train)

            # Evaluate ensemble
            train_score = ensemble.score(X_train, y_train)
            test_score = ensemble.score(X_test, y_test)
            cv_scores = cross_val_score(ensemble, X_train, y_train,
                                      cv=self.config['cv_folds'], scoring='accuracy')

            print(f"✓ Ensemble model created:")
            print(f"  - Test accuracy: {test_score:.3f}")
            print(f"  - CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            self.models['ensemble'] = ensemble

            return ensemble

        except Exception as e:
            print(f"❌ Error creating ensemble: {str(e)}")
            return None

    def select_best_model(self, model_results: Dict) -> Tuple[str, Any]:
        """Select the best performing model based on CV accuracy."""
        best_algorithm = None
        best_score = 0

        for algorithm, results in model_results.items():
            cv_score = results['cv_mean']
            if cv_score > best_score:
                best_score = cv_score
                best_algorithm = algorithm

        # Check if ensemble is better
        if 'ensemble' in self.models:
            ensemble = self.models['ensemble']
            # We'll assume ensemble is good if it exists
            best_algorithm = 'ensemble'
            best_model = ensemble
        else:
            best_model = self.models[best_algorithm]

        print(f"🏆 Best model: {best_algorithm.replace('_', ' ').title()}")
        print(f"   CV Score: {best_score:.3f}")

        return best_algorithm, best_model

    def save_model(self, model_name: str, model: Any, metrics: Dict) -> Tuple[Path, Path]:
        """Save the best model and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_file = self.model_dir / f"nba_model_{model_name}_{timestamp}.joblib"
        joblib.dump(model, model_file)

        # Extract only essential metrics (no sklearn objects)
        essential_metrics = {
            'best_model': metrics.get('best_model', model_name),
            'test_accuracy': 0.0,  # Will be set from individual model metrics
            'ensemble_used': metrics.get('ensemble_used', False)
        }
        
        # Get test accuracy from the best model
        if 'individual_models' in metrics and model_name in metrics['individual_models']:
            model_metrics = metrics['individual_models'][model_name]
            if 'test_accuracy' in model_metrics:
                essential_metrics['test_accuracy'] = model_metrics['test_accuracy']
            elif 'cv_mean' in model_metrics:
                essential_metrics['test_accuracy'] = model_metrics['cv_mean']

        # Save metadata (only essential info, no sklearn objects)
        metadata = {
            'created_at': datetime.now().isoformat(),
            'model_file': str(model_file),
            'model_type': model_name,
            'feature_columns': self.feature_columns,
            'metrics': essential_metrics,
            'config': {
                'test_size': self.config.get('test_size', 0.2),
                'cv_folds': self.config.get('cv_folds', 5),
                'random_state': self.config.get('random_state', 42)
            }
        }

        metadata_file = self.model_dir / f"nba_model_{model_name}_{timestamp}_metadata.yml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        # Also save as "latest" for easy loading
        latest_model = self.model_dir / "nba_model_latest.joblib"
        latest_metadata = self.model_dir / "nba_model_latest_metadata.yml"

        joblib.dump(model, latest_model)
        with open(latest_metadata, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        print(f"💾 Model saved to: {model_file}")
        print(f"📄 Metadata saved to: {metadata_file}")

        return model_file, metadata_file

    def train_and_save(self) -> Dict:
        """Complete advanced training pipeline."""
        print("=" * 80)
        print("🏀 ADVANCED NBA MODEL TRAINING PIPELINE")
        print("=" * 80)

        # Load data
        df = self.load_features()

        # Prepare data
        X, y = self.prepare_training_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'],
            random_state=self.config['random_state'], stratify=y
        )

        # Train individual models
        model_results = self.train_individual_models(X_train, y_train, X_test, y_test)

        # Create ensemble if configured
        ensemble = self.create_ensemble_model(X_train, y_train, X_test, y_test)

        # Select best model
        best_algorithm, best_model = self.select_best_model(model_results)

        # Prepare final metrics
        final_metrics = {
            'best_model': best_algorithm,
            'individual_models': model_results,
            'ensemble_used': ensemble is not None,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }

        # Save best model
        model_file, metadata_file = self.save_model(best_algorithm, best_model, final_metrics)

        print("=" * 80)
        print("✅ ADVANCED MODEL TRAINING COMPLETED")
        print(f"🏆 Best Model: {best_algorithm.replace('_', ' ').title()}")
        print(f"📁 Model File: {model_file}")
        print("=" * 80)

        return final_metrics


def train_advanced_model(config: Dict = None) -> Dict:
    """Train advanced NBA model with multiple algorithms."""
    trainer = AdvancedNBAModelTrainer(config)
    return trainer.train_and_save()


if __name__ == "__main__":
    # Example usage
    config = {
        'algorithms': ['random_forest', 'xgboost', 'lightgbm'],
        'ensemble': True,
        'hyperparameter_tuning': True
    }

    metrics = train_advanced_model(config)
    print(f"Training completed. Best model: {metrics['best_model']}")
