"""Unit tests for manual update script behavior."""

import unittest
from datetime import date as real_date
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.manual_update import ManualUpdater


class AprilDate:
    """Mock date class for an in-season date (Jan-Jun)."""

    @classmethod
    def today(cls):
        return real_date(2026, 4, 4)


class NovemberDate:
    """Mock date class for an in-season date (Oct-Dec)."""

    @classmethod
    def today(cls):
        return real_date(2026, 11, 15)


class TestManualUpdaterDefaults(unittest.TestCase):
    """Tests for ManualUpdater default year selection."""

    @patch("scripts.manual_update.date", AprilDate)
    def test_default_years_april_use_previous_and_current_year(self):
        updater = ManualUpdater()
        self.assertEqual(updater.start_year, 2025)
        self.assertEqual(updater.end_year, 2026)

    @patch("scripts.automation_base.date", NovemberDate)
    @patch("scripts.manual_update.date", NovemberDate)
    def test_default_years_november_use_current_and_next_year(self):
        updater = ManualUpdater()
        self.assertEqual(updater.start_year, 2026)
        self.assertEqual(updater.end_year, 2027)

    @patch("scripts.manual_update.date", AprilDate)
    def test_explicit_years_override_defaults(self):
        updater = ManualUpdater(start_year=2023, end_year=2024)
        self.assertEqual(updater.start_year, 2023)
        self.assertEqual(updater.end_year, 2024)


if __name__ == "__main__":
    unittest.main()
