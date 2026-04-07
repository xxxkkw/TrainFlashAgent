import unittest
import tempfile
import os
import numpy as np
from src.trainflashagent.verification import VerificationSuite


class TestVerificationSuite(unittest.TestCase):
    def setUp(self):
        self.verifier = VerificationSuite()

    def test_verify_model_fidelity_preserved(self):
        """Test when model fidelity is preserved (losses are similar)"""
        baseline_logs = [1.0, 0.9, 0.8, 0.7, 0.6]
        sandbox_logs = [1.0001, 0.9001, 0.8001, 0.7001, 0.6001]

        result = self.verifier.verify_model_fidelity(baseline_logs, sandbox_logs, tolerance=1e-3)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_preserved"])
        self.assertLess(result["mean_diff"], 1e-3)

    def test_verify_model_fidelity_not_preserved(self):
        """Test when model fidelity is NOT preserved (losses differ significantly)"""
        baseline_logs = [1.0, 0.9, 0.8, 0.7, 0.6]
        sandbox_logs = [1.5, 0.5, 0.3, 0.2, 0.1]

        result = self.verifier.verify_model_fidelity(baseline_logs, sandbox_logs, tolerance=1e-3)

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["is_preserved"])

    def test_verify_model_fidelity_different_lengths(self):
        """Test alignment of different length logs"""
        baseline_logs = [1.0, 0.9, 0.8, 0.7, 0.6]
        sandbox_logs = [1.0001, 0.9001, 0.8001]  # Shorter

        result = self.verifier.verify_model_fidelity(baseline_logs, sandbox_logs, tolerance=1e-3)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_preserved"])

    def test_verify_gradient_fidelity_preserved(self):
        """Test when gradient fidelity is preserved"""
        baseline_grads = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sandbox_grads = np.array([[1.0001, 2.0001, 3.0001], [4.0001, 5.0001, 6.0001]])

        result = self.verifier.verify_gradient_fidelity(baseline_grads, sandbox_grads, tolerance=1e-4)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_preserved"])
        self.assertGreater(result["cosine_similarity"], 0.9999)

    def test_verify_gradient_fidelity_not_preserved(self):
        """Test when gradient fidelity is NOT preserved"""
        baseline_grads = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        sandbox_grads = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # Orthogonal

        result = self.verifier.verify_gradient_fidelity(baseline_grads, sandbox_grads, tolerance=1e-4)

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["is_preserved"])

    def test_benchmark_throughput_no_script(self):
        """Test benchmark with non-existent script"""
        result = self.verifier.benchmark_throughput("/nonexistent/path.py", steps=10)

        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
