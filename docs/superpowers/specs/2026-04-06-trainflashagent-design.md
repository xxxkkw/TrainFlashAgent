# Design Spec: TrainFlashAgent
**Date:** 2026-04-06
**Topic:** AI-driven Top-Down Training Performance Optimizer
**Status:** Draft

## 1. Project Vision
`TrainFlashAgent` is not a simple tuning script, but a set of **Agent-native Skills** that empower an LLM to act as a Senior DL Performance Engineer. It implements a **Top-Down Diagnostic Methodology** to identify and resolve training bottlenecks in a safe, sandboxed environment, prioritizing high-impact engineering fixes over low-impact kernel tuning.

### Core Goals:
- **Maximize Throughput**: Increase training speed without compromising model accuracy.
- **Engineering First**: Prioritize solving IO bottlenecks, data skew/long-tail issues, and synchronization overhead.
- **Safe Iteration**: All experiments occur in a physically isolated sandbox with strict verification before merging.
- **Top-Down Logic**: Move from macro-timers $\rightarrow$ mid-level engineering fixes $\rightarrow$ micro-profilers.

---

## 2. The Top-Down Methodology
The Agent must follow a strict hierarchy of diagnosis and intervention to avoid "local optima" and "profiler noise."

### Phase 1: Macro-Diagnosis (Top)
- **Action**: Inject manual timers (`time.perf_counter`) at key boundaries: `DataLoader` $\rightarrow$ `Forward` $\rightarrow$ `Backward` $\rightarrow$ `Optimizer Step`.
- **Goal**: Identify which major stage is the bottleneck and analyze time variance (detecting long-tail samples).
- **Success Criteria**: A clear identification of the "slowest stage" with quantified variance.
- **Exit Condition**: Progression to Phase 2 is only permitted when the Agent can prove that macro-level tuning (e.g., simple timer-based fixes) has reached a plateau or the bottleneck has demonstrably shifted to a more granular level.

### Phase 2: Mid-Level Engineering Tuning (Middle)
- **Action**: Implement engineering fixes based on Phase 1.
    - **IO/Data**: Adjust `num_workers`, `prefetch_factor`, data bundling, or shuffling strategies to fix skew.
    - **Resources**: Optimize memory layout, Batch Size, or Gradient Accumulation.
    - **Framework**: Reduce CPU-GPU syncs or redundant Python overhead.
    - **Equivalent Substitutions**: Replace inefficient code patterns with equivalent, higher-performance implementations (e.g., vectorization).
- **Goal**: Resolve the macro-bottleneck through combinatorial testing to find the local optimum for the specific hardware and code structure.
- **Exit Condition**: Progression to Phase 3 is only permitted when engineering-level optimizations have reached their limit and the remaining bottleneck requires micro-level kernel analysis.

### Phase 3: Micro-Diagnosis (Bottom)
- **Action**: Invoke heavy profilers (`torch.profiler`, NVIDIA Nsight).
- **Goal**: Analyze specific kernel execution times or memory bandwidth.
- **Constraint**: This is the **last resort** and only used if Phase 2 yields diminishing returns.
ction**: Invoke heavy profilers (`torch.profiler`, NVIDIA Nsight).
- **Goal**: Analyze specific kernel execution times or memory bandwidth.
- **Constraint**: This is the **last resort** and only used if Phase 2 yields diminishing returns.

---

## 3. System Architecture & Skills

### 3.1 Sandbox Manager (The Safety Net)
Ensures the original codebase remains untouched until the final merge.
- `sandbox_clone(src, dst)`: Deep copy of the project.
- `sandbox_snapshot(label)`: Save state before a new optimization attempt.
- `sandbox_rollback(label)`: Revert to a known working state.
- `merge_to_main(sandbox, original)`: Safe write-back of verified changes.

### 3.2 Diagnostic Toolset (The Eyes)
- `inject_manual_timer(path, position, label)`: Inserts simple timing code into the Python source.
- `collect_timer_logs(path, steps)`: Runs training for $N$ steps (User-configurable, default $\ge 50$) and aggregates timings.
- `analyze_time_variance(logs)`: Calculates mean and variance to detect data long-tail issues.
- `run_detailed_profiler(path)`: Executes a full-scale profiler run for micro-analysis.

### 3.3 Engineering Intervention (The Hands)
- `modify_data_pipeline(path, strategy, params)`: Updates data loading/preprocessing logic.
- `tune_resource_config(path, params)`: Updates hyper-parameters related to hardware utilization.
- `edit_code(path, search_pattern, replacement)`: General purpose code modification for custom engineering fixes.

### 3.4 Verification Suite (The Judge)
- `benchmark_throughput(path, steps)`: Measures samples/sec or tokens/sec.
- `verify_model_fidelity(baseline, sandbox, steps)`: 
    - Compares Loss curves.
    - Checks gradient distribution (KL divergence) between original and optimized versions.
    - **Requirement**: The optimized version must match the baseline within a strict tolerance.

---

## 4. The Agent Decision Loop & Governance

### 4.1 Progression & Gate Logic
The Agent must not skip phases. Transition between phases requires a "Progression Proof":
- **Gate**: Before moving from Phase $N \rightarrow N+1$, the Agent must document that the current phase's optimization curve has flattened.
- **Combinatorial Search**: In each phase, the Agent must perform a set of combined tests (e.g., adjusting both `num_workers` and `prefetch_factor` simultaneously) to find the optimal configuration for the current hardware/code structure.

### 4.2 Change Tracking & Audit Log
Every modification must be recorded in a structured `Tuning Log`:
- **Modification**: Exact code change or parameter update.
- **Expected Gain**: Rationale for the change.
- **Measured Gain**: $\Delta$ Throughput / $\Delta$ Variance after verification.
- **Status**: `Pending` $\rightarrow$ `Verified` $\rightarrow$ `Merged`.

### 4.3 Reporting & Approval Workflow
- **Gain Report**: Whenever a modification yields a positive result, the Agent must report: *"Modified X $\rightarrow$ achieved Y% speedup. Requesting permission to commit to sandbox baseline."*
- **Approval**: 
    - **Explicit**: User grants permission.
    - **Implicit**: User has configured "Default Allow" for gains above a certain threshold.
- **Merge**: Only upon approval (or default allow) is the change committed to the sandbox baseline.

### 4.4 Execution Loop
1. **Clone**: `sandbox_clone` $\rightarrow$ Create environment.
2. **Macro-Probe**: `inject_manual_timer` $\rightarrow$ `collect_timer_logs` $\rightarrow$ `analyze_time_variance`.
3. **Hypothesize**: "The bottleneck is in data loading due to long-tail sample processing."
4. **Intervene**: `modify_data_pipeline` $\rightarrow$ Apply fix.
5. **Verify**: `benchmark_throughput` $\rightarrow$ `verify_model_fidelity`.
6. **Report & Approve**: Report gain $\rightarrow$ Get approval $\rightarrow$ Update baseline.
7. **Iterate**: Repeat within phase until plateau $\rightarrow$ Trigger Gate $\rightarrow$ Next Phase.
8. **Commit**: `merge_to_main`.

---

## 5. GitHub Growth Strategy (Positioning)

**Positioning Statement**: *"The first AI Agent that thinks like a DL Performance Engineer."*

**Key Hooks for Stars**:
- **Methodology as a Feature**: Market it as a "Top-Down" implementation, not just a tool.
- **Agent-Native/MCP**: Support MCP for direct integration into Cursor/Claude Desktop.
- **Case Study Focus**: Show "Before vs After" not just in speed, but in "Decision Path" (e.g., "Agent found data skew $\rightarrow$ fixed bundling $\rightarrow$ $3\times$ speedup").
- **Zero-Risk Guarantee**: Highlight the Sandbox $\rightarrow$ Fidelity Check $\rightarrow$ Merge workflow.
