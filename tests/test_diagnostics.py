import unittest
import ast
import os
import subprocess
import tempfile
import shutil
from src.trainflashagent.diagnostics import TimerInjector, collect_timer_logs, analyze_time_variance

class TestDiagnostics(unittest.TestCase):
    def test_inject_manual_timer_basic(self):
        code = "def train_step():\n    print('training')\n    return True"
        target_func = "train_step"
        label = "StepTime"

        injector = TimerInjector()
        modified_code = injector.inject_manual_timer(code, target_func, label)

        self.assertIn("time.perf_counter()", modified_code)
        self.assertIn(f"[TRAINOPT_TIMER] {label}:", modified_code)

        # Verify it's actually executable and works
        exec_globals = {}
        exec(modified_code, exec_globals)

        # Capture stdout to see if timer prints
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            exec_globals["train_step"]()

        output = f.getvalue()
        self.assertIn(f"[TRAINOPT_TIMER] {label}:", output)

    def test_analyze_time_variance(self):
        logs = [
            {'label': 'Forward', 'time': 0.1},
            {'label': 'Forward', 'time': 0.11},
            {'label': 'Forward', 'time': 0.12},
            {'label': 'Forward', 'time': 0.5}, # Long tail
            {'label': 'Backward', 'time': 0.2},
            {'label': 'Backward', 'time': 0.21},
        ]

        report = analyze_time_variance(logs)

        self.assertIn('Forward', report)
        self.assertIn('Backward', report)

        forward_stats = report['Forward']
        self.assertAlmostEqual(forward_stats['mean'], (0.1 + 0.11 + 0.12 + 0.5) / 4)
        self.assertEqual(forward_stats['max'], 0.5)
        self.assertTrue(forward_stats['is_long_tail'])

    def test_collect_timer_logs(self):
        # Create a temporary python script that prints the timer logs
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "train.py")
            with open(script_path, "w") as f:
                f.write("import time\nfor i in range(5):\n    print(f'[TRAINOPT_TIMER] Forward: {0.1 * (i+1)}')\n")

            logs = collect_timer_logs(script_path, steps=5)

            self.assertEqual(len(logs), 5)
            self.assertEqual(logs[0]['label'], 'Forward')
            self.assertAlmostEqual(logs[0]['time'], 0.1)

if __name__ == "__main__":
    unittest.main()
