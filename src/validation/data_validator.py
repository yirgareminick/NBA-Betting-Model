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