# Skill 01 — Sandbox (Isolate Before You Touch Anything)

## Purpose
- Create a fully isolated workspace for all experiments before changing any source code.

## Scope
- Applies to: any training project where code/config changes may impact correctness, reproducibility, or model quality.
- Not in scope: “quick edits” on the original project directory.

## Contract
**Inputs**
- `PROJECT_ROOT`: the path to the original training project.
- A sandbox base directory you can safely write to (e.g., `/tmp`, a scratch disk).

**Outputs**
- `SANDBOX_ROOT`: a cloned project directory you will use for all edits and runs.
- A short “Sandbox Report” that records what was cloned and how to reproduce the sandbox.

## Guardrails (MUST)
- MUST NOT modify any files under `PROJECT_ROOT`.
- MUST exclude VCS metadata and transient artifacts from the clone (`.git/`, caches, large checkpoints unless required).
- MUST keep sandbox changes reviewable (prefer a clean diff against the original).

## Procedure
### Step 1 — Choose paths
- Set `PROJECT_ROOT` to the original project directory.
- Derive a stable sandbox directory name (project name + date or a short hash).

Example:
```bash
PROJECT_ROOT="/path/to/training/project"
SANDBOX_ROOT="/tmp/trainflashagent_sandboxes/<project_name>"
```

### Step 2 — Clone into the sandbox
Preferred (keeps permissions and is selective):
```bash
mkdir -p "$SANDBOX_ROOT"
rsync -a \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache" \
  --exclude ".mypy_cache" \
  --exclude ".venv" \
  "$PROJECT_ROOT/" "$SANDBOX_ROOT/"
```

If `rsync` is unavailable, use a full copy and then remove transient files.

### Step 3 — Sanity check the clone
- Confirm the entry point and configuration files exist.
- Confirm the sandbox can run a minimal command (import checks or a dry-run flag if available).

Example:
```bash
ls "$SANDBOX_ROOT"
```

### Step 4 — Define “writeback” policy upfront
- Decide how changes will be transferred back:
  - Option A: copy a small set of files (recommended for minimal risk).
  - Option B: apply a patch / PR-style diff.
  - Option C: sync the full directory (only if you can guarantee exclusions and review).

## Definition of Done
- [ ] Sandbox directory exists and contains a runnable clone of the project.
- [ ] No files in `PROJECT_ROOT` were modified.
- [ ] A writeback method is chosen and documented.
- [ ] Sandbox Report is produced.

## Sandbox Report (Template)
```
[Sandbox Report]
Project Root: <PROJECT_ROOT>
Sandbox Root: <SANDBOX_ROOT>
Clone Method: <rsync|cp|other>
Excluded Paths: <.git, caches, ...>
Entry Point: <train.py / launcher command>
Writeback Plan: <copy files|patch|rsync>
Notes: <anything special about env/data>
```
