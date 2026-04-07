import shutil
import os
from datetime import datetime

class SandboxManager:
    def __init__(self, snapshot_base=None):
        self.snapshot_base = snapshot_base
        if self.snapshot_base:
            os.makedirs(self.snapshot_base, exist_ok=True)

    def sandbox_clone(self, src, dst):
        """
        Deep copy of the source project to the destination.
        If dst exists, it is cleared first.
        """
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def sandbox_snapshot(self, current_sandbox, label=None):
        """
        Create a timestamped or labeled backup of the current sandbox state.
        """
        if not self.snapshot_base:
            raise ValueError("snapshot_base must be configured for snapshots")

        if label is None:
            label = datetime.now().strftime("%Y%m%d_%H%M%S")

        snapshot_path = os.path.join(self.snapshot_base, label)
        if os.path.exists(snapshot_path):
            shutil.rmtree(snapshot_path)

        shutil.copytree(current_sandbox, snapshot_path)
        return snapshot_path

    def sandbox_rollback(self, current_sandbox, label):
        """
        Restore the sandbox to a previous snapshot.
        """
        if not self.snapshot_base:
            raise ValueError("snapshot_base must be configured for rollbacks")

        snapshot_path = os.path.join(self.snapshot_base, label)
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Snapshot {label} not found")

        if os.path.exists(current_sandbox):
            shutil.rmtree(current_sandbox)

        shutil.copytree(snapshot_path, current_sandbox)

    def merge_to_main(self, sandbox, original):
        """
        Selectively copy modified files from sandbox back to original.
        A simple implementation: copy all files from sandbox to original
        that are different or new.
        """
        for root, dirs, files in os.walk(sandbox):
            rel_path = os.path.relpath(root, sandbox)
            dest_root = os.path.join(original, rel_path)

            os.makedirs(dest_root, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_root, file)

                # Check if file is different or doesn't exist in original
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                else:
                    # Basic comparison: check size and mtime, or just copy
                    # For a more robust check, we could use hashing
                    with open(src_file, 'rb') as fsrc, open(dst_file, 'rb') as fdst:
                        if fsrc.read() != fdst.read():
                            shutil.copy2(src_file, dst_file)
