"""Unit tests for AutomationBase.finalize behavior."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.automation_base import AutomationBase


class DummyAutomation(AutomationBase):
    """Small concrete subclass for testing the base automation behavior."""

    def __init__(self):
        super().__init__("dummy_automation")

    def check_data_quality(self):
        return {"feature_records": "missing", "latest_date": "missing"}


class TestAutomationBaseFinalize(unittest.TestCase):
    """Ensure finalize handles notification payloads safely."""

    def test_finalize_handles_missing_feature_counts(self):
        automation = DummyAutomation()

        with mock.patch.object(automation, "cleanup_old_logs") as mock_cleanup, \
            mock.patch.object(automation, "send_notification") as mock_send:
            automation.finalize(success=True, send_notification=True)

            mock_cleanup.assert_called_once()
            mock_send.assert_called_once()
            message = mock_send.call_args.args[0]
            self.assertIn("Records: missing", message)
            self.assertIn("Latest: missing", message)


if __name__ == "__main__":
    unittest.main()
