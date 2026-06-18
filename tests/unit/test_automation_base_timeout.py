"""Test timeout handling in AutomationBase.run_command/run_python_script."""

import subprocess
import logging
import sys
from pathlib import Path
import unittest
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.automation_base import AutomationBase


class DummyAutomation(AutomationBase):
    def __init__(self):
        # Avoid running AutomationBase.__init__ which runs subprocess checks
        self.script_name = "dummy"
        self.project_root = Path(__file__).parent.parent
        self.log_dir = self.project_root / "logs"
        self.log_file = self.log_dir / "dummy.log"
        self.logger = logging.getLogger("dummy_test")
        # Ensure logger has at least one handler
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
        self.python_cmd = ["python"]


class TestAutomationTimeout(unittest.TestCase):
    def test_run_python_script_timeout_allow_failure(self):
        dummy = DummyAutomation()

        # Make subprocess.run raise TimeoutExpired
        timeout_error = subprocess.TimeoutExpired(
            cmd="cmd",
            timeout=1,
            output="partial out",
            stderr="partial err",
        )
        with mock.patch("subprocess.run", side_effect=timeout_error):
            with mock.patch.object(dummy.logger, "error") as mock_error:
                result = dummy.run_python_script("script.py", args=[], allow_failure=True, timeout=1)
            self.assertIsInstance(result, subprocess.CompletedProcess)
            self.assertEqual(result.returncode, 124)
            self.assertEqual(result.args, ["python", "script.py"])
            self.assertEqual(result.stdout, "partial out")
            self.assertEqual(result.stderr, "partial err")
            mock_error.assert_any_call("Timeout: Running script.py (after 1s)")


if __name__ == "__main__":
    unittest.main()
