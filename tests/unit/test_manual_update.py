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


class TestManualUpdaterValidation(unittest.TestCase):
    """Tests for ManualUpdater configuration validation."""

    def test_mutually_exclusive_flags_raises_error(self):
        """Ensure only one update flag can be set at a time."""
        with self.assertRaises(ValueError) as context:
            ManualUpdater(odds_only=True, games_only=True)
        self.assertIn("only one of", str(context.exception).lower())

    def test_three_flags_raises_error(self):
        """Ensure setting three flags also raises error."""
        with self.assertRaises(ValueError):
            ManualUpdater(odds_only=True, games_only=True, teams_only=True)

    def test_valid_single_flag_does_not_raise(self):
        """Ensure a single flag does not raise an error."""
        updater = ManualUpdater(odds_only=True)
        self.assertTrue(updater.odds_only)
        self.assertFalse(updater.games_only)


if __name__ == "__main__":
    unittest.main()
