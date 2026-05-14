"""Unit test for manual_update dry-run behavior."""

import unittest
from pathlib import Path
import sys
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.manual_update import ManualUpdater


class TestManualUpdaterDryRun(unittest.TestCase):
    """Ensure dry-run returns success and logs a clear completion message."""

    def test_dry_run_logs_message(self):
        updater = ManualUpdater()
        with mock.patch.object(updater.logger, "info") as mock_info:
            result = updater.run(dry_run=True)
            self.assertTrue(result)
            mock_info.assert_any_call(
                "Dry run complete: no files were changed and no external jobs were executed"
            )


if __name__ == "__main__":
    unittest.main()
