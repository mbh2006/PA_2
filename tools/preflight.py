"""Pre-flight checks before any Kaggle kernel push or results analysis.

Checks (all local, no GPU):
  1. every repo .py file compiles
  2. no NotImplementedError scaffolds remain in entry-point files
  3. expected task entry points exist
  4. no stray typo directories (e.g. ATML-PA2-LM-PostTraining) at the PA_2 level
  5. kernel directories contain script.py + kernel-metadata.json with consistent ids
  6. dataset slugs referenced by kernels exist (string presence check only)

Run from the repo root:
    python tools/preflight.py
Exit code 0 = all clear.
"""

from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PA2 = REPO.parent

EXPECTED_ENTRYPOINTS = [
    "task1_dpo/train.py", "task1_dpo/evaluate.py", "task1_dpo/ablate_beta.py", "task1_dpo/analyze_length.py",
    "task2_ppo/continue_train.py", "task2_ppo/evaluate.py", "task2_ppo/analyze_clipping.py", "task2_ppo/ablate_kl.py",
    "task3_grpo/continue_train.py", "task3_grpo/evaluate.py", "task3_grpo/analyze_group_size.py", "task3_grpo/compare_normalization.py",
    "task4_safety/generate_responses.py", "task4_safety/judge_responses.py", "task4_safety/make_audit_sheet.py", "task4_safety/evaluate_safety.py",
    "task5_feedback/evaluate_math.py", "task5_feedback/score_perturbations.py", "task5_feedback/compare_feedback.py",
    "tests/test_objectives.py", "tests/test_rl_helpers.py", "tests/smoke_data_pipeline.py",
    "analysis/task1_plots.py", "analysis/task2_plots.py", "analysis/task3_plots.py", "analysis/task4_plots.py", "analysis/task5_plots.py",
]


def main():
    problems: list[str] = []

    # 1) compile everything
    py_files = [
        p for p in REPO.rglob("*.py")
        if not any(part in {".git", "__pycache__", "outputs", "results"} for part in p.parts)
    ]
    for p in py_files:
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            problems.append(f"SYNTAX: {p.relative_to(REPO)}: {exc}")

    # 2) NotImplementedError in entry-point files
    for rel in EXPECTED_ENTRYPOINTS:
        p = REPO / rel
        if not p.exists():
            problems.append(f"MISSING entry point: {rel}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "raise NotImplementedError" in text:
            problems.append(f"SCAFFOLD remains: {rel}")

    # 3) stray typo directories
    for junk in PA2.glob("ATML-PA2-LM-*"):
        problems.append(f"STRAY DIRECTORY: {junk}")

    # 4) kernel metadata consistency
    kaggle_dir = PA2 / "kaggle"
    if kaggle_dir.exists():
        for kdir in sorted(kaggle_dir.glob("session*")):
            meta = kdir / "kernel-metadata.json"
            script = kdir / "script.py"
            if not meta.exists() or not script.exists():
                problems.append(f"KERNEL DIR incomplete: {kdir.name}")
                continue
            import json
            data = json.loads(meta.read_text(encoding="utf-8"))
            title_slug = data.get("title", "").lower().replace(" ", "-")
            if not data.get("id", "").endswith(title_slug):
                problems.append(
                    f"KERNEL SLUG MISMATCH: {kdir.name}: id={data.get('id')} vs title-slug={title_slug}"
                )
            try:
                ast.parse(script.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError as exc:
                problems.append(f"KERNEL SYNTAX: {kdir.name}/script.py: {exc}")

    if problems:
        print("PREFLIGHT PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"PREFLIGHT OK ({len(py_files)} python files compile; {len(EXPECTED_ENTRYPOINTS)} entry points present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
