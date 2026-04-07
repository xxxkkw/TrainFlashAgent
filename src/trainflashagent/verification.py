import subprocess
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple

class VerificationSuite:
    """
    The 'Judge' of the optimization loop.
    Responsible for measuring actual speed gains and ensuring model fidelity is preserved.
    """

    def benchmark_throughput(self, sandbox_path: str, steps: int = 100) -> Dict[str, Any]:
        """
        Measures the training throughput (samples/sec).
        Expects the script to print throughput in the format: [TRAINOPT_THROUGHPUT] X.XX
        """
        try:
            # We run the script. In a real scenario, we'd pass --benchmark_steps
            result = subprocess.run(
                ["python3", sandbox_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            output = result.stdout
        except subprocess.TimeoutExpired:
            output = ""

        # Pattern: [TRAINOPT_THROUGHPUT] 123.45
        pattern = r"\[TRAINOPT_THROUGHPUT\]\s+([0-9.]+)"
        matches = re.findall(pattern, output)

        if not matches:
            return {"status": "error", "message": "No throughput logs found"}

        throughputs = [float(m) for m in matches]
        avg_throughput = sum(throughputs) / len(throughputs)

        return {
            "status": "success",
            "avg_throughput": avg_throughput,
            "max_throughput": max(throughputs),
            "min_throughput": min(throughputs),
            "samples": len(throughputs)
        }

    def verify_model_fidelity(self, baseline_logs: List[float], sandbox_logs: List[float], tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Compares the loss curves of the baseline and the optimized version.

        Args:
            baseline_logs: List of loss values from original run.
            sandbox_logs: List of loss values from optimized run.
            tolerance: Absolute tolerance for loss difference.
        """
        if len(baseline_logs) != len(sandbox_logs):
            # Try to align lengths
            min_len = min(len(baseline_logs), len(sandbox_logs))
            baseline_logs = baseline_logs[:min_len]
            sandbox_logs = sandbox_logs[:min_len]

        diffs = np.abs(np.array(baseline_logs) - np.array(sandbox_logs))
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)

        # Fidelity is preserved if the mean difference is within tolerance
        is_preserved = mean_diff < tolerance

        return {
            "status": "success",
            "is_preserved": is_preserved,
            "mean_diff": float(mean_diff),
            "max_diff": float(max_diff),
            "tolerance": tolerance
        }

    def verify_gradient_fidelity(self, baseline_grads: np.ndarray, sandbox_grads: np.ndarray, tolerance: float = 1e-4) -> Dict[str, Any]:
        """
        Checks the gradient distribution (using cosine similarity or KL divergence)
        between original and optimized versions.
        """
        # Flatten arrays for comparison
        b_flat = baseline_grads.flatten()
        s_flat = sandbox_grads.flatten()

        # Cosine Similarity
        cos_sim = np.dot(b_flat, s_flat) / (np.linalg.norm(b_flat) * np.linalg.norm(s_flat))

        # Fidelity is preserved if similarity is very close to 1.0
        is_preserved = cos_sim > (1.0 - tolerance)

        return {
            "status": "success",
            "is_preserved": is_preserved,
            "cosine_similarity": float(cos_sim),
            "tolerance": tolerance
        }
