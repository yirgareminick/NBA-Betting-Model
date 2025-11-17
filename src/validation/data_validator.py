"""
Data Validation Framework for NBA Betting Model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    count: Optional[int] = None


class DataValidator:
    """Data validation for NBA pipeline."""

    def __init__(self):
        self.valid_teams = ['LAL', 'BOS', 'GSW', 'MIA', 'CHI']

    def validate_games_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate games data."""
        results = []
        required_cols = ['game_id', 'game_date', 'home_team', 'away_team']

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing columns: {missing_cols}"
            ))

        # Check for duplicates
        if 'game_id' in df.columns:
            duplicates = df.duplicated(subset=['game_id']).sum()
            if duplicates > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Found {duplicates} duplicate games",
                    count=duplicates
                ))

        # Validate team names
        for team_col in ['home_team', 'away_team']:
            if team_col in df.columns:
                invalid_teams = ~df[team_col].isin(self.valid_teams)
                invalid_count = invalid_teams.sum()
                if invalid_count > 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Invalid {team_col} names found",
                        count=invalid_count
                    ))

        # Check date validity
        if 'game_date' in df.columns:
            try:
                dates = pd.to_datetime(df['game_date'])
                invalid_dates = dates.isnull().sum()
                if invalid_dates > 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Invalid dates found",
                        count=invalid_dates
                    ))
            except Exception:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Date parsing failed"
                ))

        return results

    def validate_features_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate features data."""
        results = []
        required_cols = ['game_id', 'team_name', 'target_win']

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing feature columns: {missing_cols}"
            ))

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Infinite values in {col}",
                    count=inf_count
                ))

        return results

    def validate_odds_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate odds data."""
        results = []
        required_cols = ['game_id', 'home_odds', 'away_odds']

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing odds columns: {missing_cols}"
            ))

        # Check odds ranges
        for odds_col in ['home_odds', 'away_odds']:
            if odds_col in df.columns:
                invalid_odds = ((df[odds_col] < 1.1) | (df[odds_col] > 10.0)).sum()
                if invalid_odds > 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Invalid odds range in {odds_col}",
                        count=invalid_odds
                    ))

        return results

    def run_validation(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[ValidationResult]]:
        """Run validation on all data types."""
        all_results = {}

        if 'games' in data_dict:
            all_results['games'] = self.validate_games_data(data_dict['games'])

        if 'features' in data_dict:
            all_results['features'] = self.validate_features_data(data_dict['features'])

        if 'odds' in data_dict:
            all_results['odds'] = self.validate_odds_data(data_dict['odds'])

        return all_results