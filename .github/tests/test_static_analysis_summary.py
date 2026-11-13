"""Run the reusable workflow's real outcome reporter against job-result fixtures."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/reusable-static-analysis.yml"
SOURCE = WORKFLOW.read_text(encoding="utf-8")
REPORTER = textwrap.dedent(
    SOURCE.split("          # STATIC_ANALYSIS_SUMMARY_BEGIN\n", 1)[1].split(
        "          # STATIC_ANALYSIS_SUMMARY_END\n", 1
    )[0]
)


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.needs = {
            "detect": {"result": "success", "outputs": {"python": "true", "docs": "true"}},
            "actions": {"result": "success", "outputs": {"syntax": "success", "security": "success"}},
            "python": {"result": "success", "outputs": {"syntax": "success", "security": "success"}},
            "docs": {"result": "success", "outputs": {"integrity": "success"}},
        }

    def report(self, enforced):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = os.environ.copy()
            env.update(ANALYSIS_NEEDS=json.dumps(self.needs),
                       ANALYSIS_ENFORCED=str(enforced).lower(),
                       GITHUB_STEP_SUMMARY=str(root / "summary.md"))
            result = subprocess.run([sys.executable, "-c", REPORTER], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertTrue((root / "static-analysis-outcomes.json").exists(), result.stderr)
            return result.returncode, json.loads((root / "static-analysis-outcomes.json").read_text()), (root / "summary.md").read_text()

    def test_success_is_recorded_in_both_modes(self):
        for enforced in (False, True):
            with self.subTest(enforced=enforced):
                code, report, summary = self.report(enforced)
                self.assertEqual(code, 0)
                self.assertEqual(report["status"], "passed")
                self.assertIn("actions/security | success", summary)

    def test_advisory_failure_remains_visible(self):
        self.needs["python"]["outputs"]["syntax"] = "failure"
        code, report, summary = self.report(False)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["mode"], "advisory")
        self.assertIn("report exits successfully in advisory mode", summary)

    def test_each_correctness_failure_blocks_enforced_mode(self):
        original = copy.deepcopy(self.needs)
        for job, name in (("actions", "syntax"), ("actions", "security"), ("python", "syntax"), ("python", "security"), ("docs", "integrity")):
            with self.subTest(job=job, name=name):
                self.needs = copy.deepcopy(original)
                self.needs[job]["outputs"][name] = "failure"
                code, report, _ = self.report(True)
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "failed")

    def test_setup_failure_and_missing_output_cannot_pass(self):
        self.needs["python"] = {"result": "failure", "outputs": {}}
        code, report, _ = self.report(True)
        self.assertEqual(code, 1)
        self.assertIn({"check": "python/syntax", "outcome": "unavailable"}, report["checks"])

    def test_no_language_is_a_legitimate_skip(self):
        self.needs["detect"]["outputs"]["python"] = "false"
        self.needs["python"] = {"result": "skipped", "outputs": {}}
        self.assertEqual(self.report(True)[0], 0)

    def test_expected_language_cannot_be_skipped(self):
        self.needs["python"] = {"result": "skipped", "outputs": {}}
        self.assertEqual(self.report(True)[0], 1)

    def test_individual_required_analyzer_cannot_be_skipped(self):
        self.needs["python"]["outputs"]["security"] = "skipped"
        self.assertEqual(self.report(True)[0], 1)

    def test_detector_failure_cannot_hide_skipped_jobs(self):
        self.needs["detect"] = {"result": "failure", "outputs": {}}
        self.needs["python"] = {"result": "skipped", "outputs": {}}
        self.needs["docs"] = {"result": "skipped", "outputs": {}}
        self.assertEqual(self.report(True)[0], 1)

    def test_cancellation_or_missing_job_is_reported_as_failure(self):
        original = copy.deepcopy(self.needs)
        for replacement in ({"result": "cancelled", "outputs": {}}, None):
            with self.subTest(replacement=replacement):
                self.needs = copy.deepcopy(original)
                if replacement is None:
                    del self.needs["actions"]
                else:
                    self.needs["actions"] = replacement
                self.assertEqual(self.report(True)[0], 1)


if __name__ == "__main__":
    unittest.main()
