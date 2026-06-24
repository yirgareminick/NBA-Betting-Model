"""Unit test for manual_update dry-run behavior."""

import unittest
from pathlib import Path
import sys
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.manual_update import ManualUpdater


class TestManualUpdaterDryRun(unittest.TestCase):
    """Ensure dry-run succeeds and logs planned actions and completion."""

    def test_dry_run_logs_message(self):
        updater = ManualUpdater()
        with mock.patch.object(updater.logger, "info") as mock_info, \
            mock.patch.object(updater, "run_python_script") as mock_run_script:
            result = updater.run(dry_run=True)
            self.assertTrue(result)

            # Ensure no scripts ran
            mock_run_script.assert_not_called()

            # Ensure a clear completion message was logged
            self.assertTrue(
                any(
                    "Dry run complete: no files were changed and no external jobs were executed" in call[0][0]
                    for call in mock_info.call_args_list
                )
            )
            self.assertEqual(
                sum(
                    "Dry run complete: no files were changed and no external jobs were executed" in call[0][0]
                    for call in mock_info.call_args_list
                ),
                1,
            )

            # Ensure planned actions were described in the dry-run
            self.assertTrue(any("Planned actions (dry-run):" in call[0][0] for call in mock_info.call_args_list))

            # Ensure the initial manual update header was logged
            self.assertTrue(
                any("Manual update (" in call[0][0] for call in mock_info.call_args_list)
            )
            self.assertTrue(
                any("Manual update (full):" in call[0][0] for call in mock_info.call_args_list)
            )
            self.assertTrue(
                any("(DRY RUN)" in call[0][0] for call in mock_info.call_args_list)
            )

    def test_dry_run_odds_only(self):
        """Ensure dry-run with --odds-only logs appropriate action."""
        updater = ManualUpdater(odds_only=True)
        with mock.patch.object(updater.logger, "info") as mock_info, \
            mock.patch.object(updater, "run_python_script"):
            result = updater.run(dry_run=True)
            self.assertTrue(result)
            self.assertTrue(
                any("Update odds for" in call[0][0] for call in mock_info.call_args_list)
            )

    def test_dry_run_games_only(self):
        """Ensure dry-run with --games-only logs appropriate action."""
        updater = ManualUpdater(games_only=True)
        with mock.patch.object(updater.logger, "info") as mock_info, \
            mock.patch.object(updater, "run_python_script"):
            result = updater.run(dry_run=True)
            self.assertTrue(result)
            self.assertTrue(
                any("Update games" in call[0][0] for call in mock_info.call_args_list)
            )

    def test_dry_run_features_only(self):
        """Ensure dry-run with --features-only logs appropriate action."""
        updater = ManualUpdater(features_only=True)
        with mock.patch.object(updater.logger, "info") as mock_info, \
            mock.patch.object(updater, "run_python_script"):
            result = updater.run(dry_run=True)
            self.assertTrue(result)
            self.assertTrue(
                any("Rebuild features" in call[0][0] for call in mock_info.call_args_list)
            )

    def test_dry_run_features_only_custom_lookback(self):
        """Ensure dry-run features-only logs the configured lookback days."""
        updater = ManualUpdater(features_only=True, lookback_days=14)
        with mock.patch.object(updater.logger, "info") as mock_info, \
            mock.patch.object(updater, "run_python_script"):
            result = updater.run(dry_run=True)
            self.assertTrue(result)
            self.assertTrue(
                any("Rebuild features with 14d lookback" in call[0][0] for call in mock_info.call_args_list)
            )

    def test_dry_run_returns_success(self):
        """Ensure dry-run always returns True (success)."""
        for odds_only, games_only, features_only in [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ]:
            with self.subTest(odds_only=odds_only, games_only=games_only, features_only=features_only):
                updater = ManualUpdater(
                    odds_only=odds_only,
                    games_only=games_only,
                    features_only=features_only
                )
                with mock.patch.object(updater.logger, "info"), \
                    mock.patch.object(updater, "run_python_script"):
                    result = updater.run(dry_run=True)
                    self.assertTrue(result, "Dry-run should always return True")

    def test_invalid_year_range_raises(self):
        """Ensure invalid year range (start > end) raises ValueError."""
        with self.assertRaisesRegex(ValueError, "cannot be greater"):
            ManualUpdater(start_year=2025, end_year=2023)

    def test_dry_run_teams_only(self):
        """Ensure dry-run with --teams-only logs appropriate action."""
        updater = ManualUpdater(teams_only=True)
        with mock.patch.object(updater.logger, "info") as mock_info, \
            mock.patch.object(updater, "run_python_script"):
            result = updater.run(dry_run=True)
            self.assertTrue(result)
            self.assertTrue(
                any("Update team stats" in call[0][0] for call in mock_info.call_args_list)
            )

    def test_dry_run_custom_bookmakers(self):
        """Ensure dry-run with custom bookmakers log them in planned actions."""
        custom_bookmakers = ["bovada", "betmgm", "caesars"]
        updater = ManualUpdater(odds_only=True, bookmakers=custom_bookmakers)
        with mock.patch.object(updater.logger, "info") as mock_info, \
            mock.patch.object(updater, "run_python_script"):
            result = updater.run(dry_run=True)
            self.assertTrue(result)
            self.assertTrue(
                any(all(bookmaker in call[0][0] for bookmaker in custom_bookmakers) for call in mock_info.call_args_list)
            )


if __name__ == "__main__":
    unittest.main()
