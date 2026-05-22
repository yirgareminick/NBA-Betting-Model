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

            # Ensure planned actions were described in the dry-run
            self.assertTrue(any("Planned actions (dry-run):" in call[0][0] for call in mock_info.call_args_list))

            # Ensure the initial manual update header was logged
            self.assertTrue(
                any("Manual update (" in call[0][0] for call in mock_info.call_args_list)
            )


if __name__ == "__main__":
    unittest.main()
