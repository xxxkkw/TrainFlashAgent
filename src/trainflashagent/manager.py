from .sandbox import SandboxManager
from .diagnostics import TimerInjector, collect_timer_logs, analyze_time_variance
from .interventions import EngineeringIntervention
from .verification import VerificationSuite
from .governance import TuningLog, ProgressionGate, Phase
import os

class TuningManager:
    """
    The Orchestrator of the TrainFlashAgent.
    Coordinates the flow between sandbox, diagnostics, interventions, and verification.
    """
    def __init__(self, project_root: str, sandbox_root: str, baseline_throughput: float = None):
        self.project_root = project_root
        self.sandbox_root = sandbox_root

        self.sandbox = SandboxManager(snapshot_base=os.path.join(sandbox_root, "snapshots"))
        self.diagnostics = TimerInjector()
        self.interventions = EngineeringIntervention()
        self.verification = VerificationSuite()
        self.log = TuningLog()
        self.gate = ProgressionGate()

        self.current_phase = Phase.MACRO_DIAGNOSTICS
        self.active_sandbox = os.path.join(sandbox_root, "active_sandbox")
        self.baseline_throughput = baseline_throughput  # Baseline for gain calculation
        self.pending_gains = []  # Pending approval gains

    def setup_environment(self):
        """Initialize the sandbox."""
        self.sandbox.sandbox_clone(self.project_root, self.active_sandbox)
        return self.active_sandbox

    def run_macro_diagnostic(self, target_file: str, target_func: str, label: str):
        """Phase 1: Macro-Probe."""
        full_path = os.path.join(self.active_sandbox, target_file)
        with open(full_path, 'r') as f:
            code = f.read()

        modified_code = self.diagnostics.inject_manual_timer(code, target_func, label)
        with open(full_path, 'w') as f:
            f.write(modified_code)

        logs = collect_timer_logs(full_path)
        report = analyze_time_variance(logs)
        return report

    def apply_and_verify_intervention(self, file_rel_path: str, intervention_type: str, params: dict = None, baseline_logs: list = None):
        """The core loop: Intervene -> Verify -> Log."""
        full_path = os.path.join(self.active_sandbox, file_rel_path)

        # 1. Snapshot for safety
        snapshot_label = self.sandbox.sandbox_snapshot(self.active_sandbox)

        # 2. Apply Intervention
        mod_idx = self.log.add_entry(f"{intervention_type} on {file_rel_path}", "Expected throughput increase")

        success = False
        if intervention_type == "data_pipeline":
            success = self.interventions.optimize_data_pipeline(full_path, params=params)
        elif intervention_type == "resource_config":
            success = self.interventions.tune_resource_config(full_path, params=params)
        elif intervention_type == "equivalent_sub":
            success = self.interventions.apply_equivalent_substitution(full_path, params.get("type", "loop_to_stack"))

        if not success:
            self.log.update_gain(mod_idx, 0.0, status="Failed/No-Change")
            return {"status": "no_change"}

        # 3. Verify Throughput
        v_result = self.verification.benchmark_throughput(full_path)
        if v_result["status"] == "success":
            current_throughput = v_result["avg_throughput"]

            # Calculate actual gain (delta from baseline)
            if self.baseline_throughput and self.baseline_throughput > 0:
                gain_delta = ((current_throughput - self.baseline_throughput) / self.baseline_throughput) * 100
            else:
                gain_delta = 0.0
                self.baseline_throughput = current_throughput  # Set baseline on first run

            self.log.update_gain(mod_idx, gain_delta, status="Verified")
            self.gate.record_gain(gain_delta)  # Record delta, not absolute value

            # 4. Fidelity Check (if baseline logs provided)
            fidelity_ok = True
            fidelity_result = None
            if baseline_logs:
                # Run optimized version and compare loss curves
                sandbox_logs = self._extract_loss_logs(full_path)
                if sandbox_logs:
                    fidelity_result = self.verification.verify_model_fidelity(baseline_logs, sandbox_logs)
                    fidelity_ok = fidelity_result["is_preserved"]

            if not fidelity_ok:
                self.sandbox.sandbox_rollback(self.active_sandbox, snapshot_label)
                self.log.update_gain(mod_idx, gain_delta, status="Rejected-Fidelity")
                return {"status": "fidelity_failed", "message": "Model fidelity not preserved", "fidelity_result": fidelity_result}

            # Add to pending gains for approval workflow
            self.pending_gains.append({
                "index": mod_idx,
                "gain": gain_delta,
                "throughput": current_throughput
            })

            return {
                "status": "success",
                "gain_percent": gain_delta,
                "throughput": current_throughput,
                "baseline_throughput": self.baseline_throughput,
                "fidelity_ok": fidelity_ok
            }

        # Rollback on failure
        self.sandbox.sandbox_rollback(self.active_sandbox, snapshot_label)
        return {"status": "error", "message": "Verification failed"}

    def _extract_loss_logs(self, script_path: str, steps: int = 10) -> list:
        """Extract loss values from training script output."""
        import subprocess
        import re
        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            # Pattern: [TRAINOPT_LOSS] X.XX
            pattern = r"\[TRAINOPT_LOSS\]\s+([0-9.]+)"
            matches = re.findall(pattern, result.stdout)
            return [float(m) for m in matches]
        except Exception:
            return []

    def get_gain_report(self) -> str:
        """Generate gain report for approval workflow (Section 4.3)."""
        if not self.pending_gains:
            return "No pending gains to report."

        report = "=== Pending Optimization Gains ===\n\n"
        for item in self.pending_gains:
            report += f"Gain: {item['gain']:+.2f}% | Throughput: {item['throughput']:.2f}\n"

        total_improvement = sum(g['gain'] for g in self.pending_gains)
        report += f"\nTotal Improvement: {total_improvement:+.2f}%\n"
        report += "\nRequesting permission to commit to sandbox baseline."
        return report

    def approve_pending_gains(self):
        """Commit pending gains to baseline after approval."""
        if self.pending_gains:
            # Update baseline to latest throughput
            latest = self.pending_gains[-1]
            self.baseline_throughput = latest['throughput']
            # Mark all as merged
            for item in self.pending_gains:
                self.log.update_gain(item['index'], item['gain'], status="Merged")
            self.pending_gains.clear()

    def check_progression(self) -> bool:
        """Check if we can move to the next phase."""
        if self.gate.can_progress():
            if self.current_phase == Phase.MACRO_DIAGNOSTICS:
                self.current_phase = Phase.ENGINEERING_TUNING
            elif self.current_phase == Phase.ENGINEERING_TUNING:
                self.current_phase = Phase.MICRO_DIAGNOSTICS
            return True
        return False

    def finalize_and_merge(self):
        """Merge verified changes back to main."""
        self.sandbox.merge_to_main(self.active_sandbox, self.project_root)
        self.current_phase = Phase.MERGED
