import unittest
import tempfile
import os
import shutil
from src.trainflashagent.manager import TuningManager
from src.trainflashagent.governance import Phase


class TestTuningManager(unittest.TestCase):
    """Integration tests for the TuningManager orchestration."""

    def setUp(self):
        """Set up temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.project_root = os.path.join(self.test_dir, "project")
        self.sandbox_root = os.path.join(self.test_dir, "sandbox")

        os.makedirs(self.project_root)

        # Create a dummy training script
        self.train_script = os.path.join(self.project_root, "train.py")
        with open(self.train_script, "w") as f:
            f.write("""
import time

def train_loop():
    for i in range(5):
        time.sleep(0.01)
        print(f"[TRAINOPT_THROUGHPUT] 100.0")
        print(f"[TRAINOPT_LOSS] {1.0 - i * 0.1}")

if __name__ == "__main__":
    train_loop()
""")

        self.manager = TuningManager(
            project_root=self.project_root,
            sandbox_root=self.sandbox_root
        )

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir)

    def test_setup_environment(self):
        """Test sandbox initialization."""
        sandbox_path = self.manager.setup_environment()

        self.assertTrue(os.path.exists(sandbox_path))
        self.assertTrue(os.path.exists(os.path.join(sandbox_path, "train.py")))

    def test_initial_phase(self):
        """Test that initial phase is MACRO_DIAGNOSTICS."""
        self.assertEqual(self.manager.current_phase, Phase.MACRO_DIAGNOSTICS)

    def test_tuning_log_initialization(self):
        """Test that tuning log is initialized."""
        self.assertEqual(len(self.manager.log.get_report()), 0)

    def test_progression_gate_initialization(self):
        """Test that progression gate is initialized."""
        self.assertFalse(self.manager.gate.can_progress())


class TestFullWorkflow(unittest.TestCase):
    """End-to-end workflow test."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_root = os.path.join(self.test_dir, "project")
        self.sandbox_root = os.path.join(self.test_dir, "sandbox")

        os.makedirs(self.project_root)

        # Create a training script with DataLoader
        self.train_script = os.path.join(self.project_root, "train.py")
        with open(self.train_script, "w") as f:
            f.write("""
import time

def train_loop():
    for i in range(3):
        time.sleep(0.01)
        print(f"[TRAINOPT_THROUGHPUT] 50.0")

if __name__ == "__main__":
    train_loop()
""")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_sandbox_clone_and_merge(self):
        """Test the full clone -> modify -> merge workflow."""
        manager = TuningManager(
            project_root=self.project_root,
            sandbox_root=self.sandbox_root
        )

        # 1. Setup sandbox
        sandbox_path = manager.setup_environment()
        self.assertTrue(os.path.exists(sandbox_path))

        # 2. Make a modification in sandbox
        sandbox_train = os.path.join(sandbox_path, "train.py")
        with open(sandbox_train, "r") as f:
            content = f.read()
        content = content.replace("50.0", "100.0")
        with open(sandbox_train, "w") as f:
            f.write(content)

        # 3. Merge back
        manager.finalize_and_merge()

        # 4. Verify original was updated
        with open(self.train_script, "r") as f:
            original_content = f.read()
        self.assertIn("100.0", original_content)


if __name__ == "__main__":
    unittest.main()
