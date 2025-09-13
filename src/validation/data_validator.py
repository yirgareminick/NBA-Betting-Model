"""
Data Validation Framework for NBA Betting Model
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
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
        
        return results