import ast
import time
import re
import subprocess
from typing import List, Dict, Any

class TimerInjector:
    """
    Analyzes and modifies Python source code to inject timing logic using AST.
    """
    def inject_manual_timer(self, code: str, target_function: str, label: str) -> str:
        tree = ast.parse(code)

        class TimerTransformer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == target_function:
                    # Start timer: start = time.perf_counter()
                    start_node = ast.parse("start = time.perf_counter()").body[0]

                    # End timer:
                    # end = time.perf_counter()
                    # print(f"[TRAINOPT_TIMER] {label}: {end - start}")
                    end_logic = ast.parse(
                        f"end = time.perf_counter()\n"
                        f"print(f'[TRAINOPT_TIMER] {label}: {{end - start}}')"
                    ).body

                    # Inject start at the beginning of the function body
                    node.body.insert(0, start_node)

                    # Inject end logic before return statements or at the end of the function
                    # To ensure timing happens even with returns, we'll wrap the body in a try-finally.

                    original_body = node.body
                    # Create a new body that consists of a try-finally block
                    # try:
                    #     [original_body]
                    # finally:
                    #     [end_logic]

                    try_node = ast.Try(
                        body=original_body,
                        handlers=[],
                        orelse=[],
                        finalbody=end_logic
                    )

                    node.body = [start_node, try_node]

                return node

        transformer = TimerTransformer()
        modified_tree = transformer.visit(tree)
        ast.fix_missing_locations(modified_tree)

        # Ensure 'import time' is present
        modified_code = ast.unparse(modified_tree)
        if "import time" not in modified_code:
            modified_code = "import time\n" + modified_code

        return modified_code

def collect_timer_logs(sandbox_path: str, steps: int = 50) -> List[Dict[str, Any]]:
    """
    Executes the training script and captures [TRAINOPT_TIMER] logs.
    """
    # In a real scenario, we might pass 'steps' as an argument to the script
    # For this implementation, we execute the script and parse the output.
    try:
        result = subprocess.run(
            ["python3", sandbox_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout
    except subprocess.TimeoutExpired as e:
        # If it times out, we might still have some logs
        output = e.stdout if e.stdout else ""

    logs = []
    # Pattern: [TRAINOPT_TIMER] Label: Value
    pattern = r"\[TRAINOPT_TIMER\]\s+(\w+):\s+([0-9.]+)"
    matches = re.findall(pattern, output)

    for label, value in matches:
        logs.append({'label': label, 'time': float(value)})

    return logs

def analyze_time_variance(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates statistics and identifies long-tail samples.
    """
    if not logs:
        return {}

    data = {}
    for log in logs:
        label = log['label']
        if label not in data:
            data[label] = []
        data[label].append(log['time'])

    report = {}
    for label, times in data.items():
        mean = sum(times) / len(times)
        max_val = max(times)
        min_val = min(times)

        # Variance/Std Dev
        variance = sum((x - mean)**2 for x in times) / len(times)
        std_dev = variance**0.5

        # Long-tail detection: if max is significantly larger than mean (e.g., > 2x)
        is_long_tail = max_val > (mean * 2) if mean > 0 else False

        report[label] = {
            'mean': mean,
            'std_dev': std_dev,
            'max': max_val,
            'min': min_val,
            'count': len(times),
            'is_long_tail': is_long_tail
        }

    return report


def run_detailed_profiler(sandbox_path: str, steps: int = 10, output_dir: str = "./profiler_output") -> Dict[str, Any]:
    """
    Executes a full-scale profiler run for micro-level analysis.
    This is the last resort, used only after Phase 2 yields diminishing returns.

    Uses torch.profiler to capture:
    - Kernel execution times
    - Memory bandwidth
    - CPU/GPU activity

    Returns a summary report and saves detailed Chrome trace.
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate profiler script that wraps the training script
    profiler_script = f"""
import torch
import sys
import os

# Add the directory of the target script to path
sys.path.insert(0, os.path.dirname(os.path.abspath('{sandbox_path}')))

# Import and run the target script with profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # Execute the training script
    exec(open('{sandbox_path}').read())

# Export Chrome trace
prof.export_chrome_trace('{output_dir}/trace.json')

# Print summary
print("\\n=== Profiler Summary ===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
"""

    # Write and execute profiler script
    profiler_file = os.path.join(output_dir, "profiler_wrapper.py")
    with open(profiler_file, "w") as f:
        f.write(profiler_script)

    try:
        result = subprocess.run(
            ["python3", profiler_file],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for profiling
        )
        output = result.stdout + "\n" + result.stderr
        trace_path = os.path.join(output_dir, "trace.json")

        return {
            "status": "success",
            "trace_path": trace_path,
            "output": output,
            "message": f"Profiler trace saved to {trace_path}"
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": "Profiler run timed out after 10 minutes"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
