import unittest
import shutil
import os
import tempfile
from src.trainflashagent.sandbox import SandboxManager

class TestSandboxManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.src_dir = os.path.join(self.test_dir, "src_project")
        self.dst_dir = os.path.join(self.test_dir, "dst_project")
        self.snapshot_dir = os.path.join(self.test_dir, "snapshots")

        os.makedirs(self.src_dir)
        with open(os.path.join(self.src_dir, "file1.txt"), "w") as f:
            f.write("content1")
        with open(os.path.join(self.src_dir, "file2.txt"), "w") as f:
            f.write("content2")

        self.manager = SandboxManager(snapshot_base=self.snapshot_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_sandbox_clone(self):
        # Test initial clone
        self.manager.sandbox_clone(self.src_dir, self.dst_dir)
        self.assertTrue(os.path.exists(self.dst_dir))
        with open(os.path.join(self.dst_dir, "file1.txt"), "r") as f:
            self.assertEqual(f.read(), "content1")

        # Test clone with existing dst (should be cleared)
        with open(os.path.join(self.dst_dir, "extra.txt"), "w") as f:
            f.write("extra")

        # Update src
        with open(os.path.join(self.src_dir, "file1.txt"), "w") as f:
            f.write("updated_content1")

        self.manager.sandbox_clone(self.src_dir, self.dst_dir)
        self.assertFalse(os.path.exists(os.path.join(self.dst_dir, "extra.txt")))
        with open(os.path.join(self.dst_dir, "file1.txt"), "r") as f:
            self.assertEqual(f.read(), "updated_content1")

    def test_snapshot_and_rollback(self):
        self.manager.sandbox_clone(self.src_dir, self.dst_dir)

        # Modify sandbox
        with open(os.path.join(self.dst_dir, "file1.txt"), "w") as f:
            f.write("modified_content")

        # Take snapshot
        label = "snap1"
        self.manager.sandbox_snapshot(self.dst_dir, label=label)

        # Modify sandbox again
        with open(os.path.join(self.dst_dir, "file1.txt"), "w") as f:
            f.write("further_modified")

        # Rollback
        self.manager.sandbox_rollback(self.dst_dir, label=label)
        with open(os.path.join(self.dst_dir, "file1.txt"), "r") as f:
            self.assertEqual(f.read(), "modified_content")

    def test_merge_to_main(self):
        self.manager.sandbox_clone(self.src_dir, self.dst_dir)

        # Modify a file in sandbox
        with open(os.path.join(self.dst_dir, "file1.txt"), "w") as f:
            f.write("merged_content")

        # Add a new file in sandbox
        with open(os.path.join(self.dst_dir, "new_file.txt"), "w") as f:
            f.write("new_file_content")

        # Merge back to src
        self.manager.merge_to_main(self.dst_dir, self.src_dir)

        with open(os.path.join(self.src_dir, "file1.txt"), "r") as f:
            self.assertEqual(f.read(), "merged_content")
        with open(os.path.join(self.src_dir, "new_file.txt"), "r") as f:
            self.assertEqual(f.read(), "new_file_content")
        # Ensure file2.txt remains unchanged
        with open(os.path.join(self.src_dir, "file2.txt"), "r") as f:
            self.assertEqual(f.read(), "content2")


if __name__ == "__main__":
    unittest.main()
