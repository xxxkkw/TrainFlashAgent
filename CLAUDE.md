# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Response Language
- Default to Chinese for chat responses unless the user explicitly asks for English.
- Keep the skills (playbooks) in English.

## Project: TrainFlashAgent
TrainFlashAgent is a set of Markdown Skills (playbooks) for improving deep learning training performance. The workflow is designed to be executed by an AI coding assistant and emphasizes measurable, auditable engineering + training-policy optimizations.

## Repository Structure
- `skills/`: the playbooks (Skills)
- `.cursorrules`: Cursor instructions
- `.github/copilot-instructions.md`: GitHub Copilot instructions
- `README.md` / `README_zh.md`: documentation

## Core Workflow (MUST follow in order)
1. Sandbox (isolate before edits)
2. Diagnose (manual timing / macro signals first)
3. Optimize (engineering + training-policy fixes)
4. Verify (performance + fidelity; convergence sanity when needed)
5. Write back only after explicit approval

## Guardrails (MUST)
- Do not skip phases.
- Keep changes reviewable and minimal; avoid unrelated refactors.
- Change one variable per experiment when benchmarking.
- Always report mean + tail (p95/max) for performance results.
- Always state the fidelity tolerance and the verification protocol.
- Do not write back to the original project without explicit user approval.

## Entry Points
- Start from `skills/01-sandbox.md` and proceed sequentially.
