"""
Evaluation harness for the Multi-Modal Evidence Review pipeline.

Runs the main pipeline against dataset/sample_claims.csv (which has expected
outputs) for one or more strategies, computes per-field accuracy against the
ground truth, measures wall-clock runtime, and writes a Markdown report.

Usage:
    python code/evaluation/main.py --strategies A B
    python code/evaluation/main.py --strategies A
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_CSV = DATASET_DIR / "sample_claims.csv"
MAIN_PY = CODE_DIR / "main.py"
REPORT_PATH = REPO_ROOT / "evaluation_report.md"

FIELDS = [
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_strategy(strategy: str) -> tuple[list[dict], float]:
    out_path = EVAL_DIR / f"eval_output_{strategy}.csv"
    start = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable, str(MAIN_PY),
            "--claims", str(SAMPLE_CSV),
            "--out", str(out_path),
            "--strategy", strategy,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"  [ERROR] strategy {strategy} exited {result.returncode}")
        print(result.stderr[-2000:])
    return read_csv(out_path), elapsed


def score(expected: list[dict], actual: list[dict]) -> dict:
    scores = {}
    n = min(len(expected), len(actual))
    for f in FIELDS:
        matches = sum(1 for e, a in zip(expected, actual) if e.get(f) == a.get(f))
        scores[f] = (matches, n)
    return scores


def format_report(strategy_results: dict) -> str:
    lines = ["# Evaluation Report\n"]
    lines.append("## Per-Field Accuracy\n")
    header = "| Field | " + " | ".join(strategy_results.keys()) + " |"
    sep = "|---|" + "---|" * len(strategy_results)
    lines.append(header)
    lines.append(sep)
    for f in FIELDS:
        row = [f]
        for strat, data in strategy_results.items():
            m, n = data["scores"][f]
            pct = (100 * m // n) if n else 0
            row.append(f"{m}/{n} ({pct}%)")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Operational Metrics\n")
    lines.append("| Strategy | Wall-clock runtime (s) | Rows processed |")
    lines.append("|---|---|---|")
    for strat, data in strategy_results.items():
        lines.append(f"| {strat} | {data['elapsed']:.1f} | {data['n']} |")

    lines.append("\n## Notes\n")
    lines.append(
        "- Per-call token usage is printed to stdout during each run.\n"
        "- Comparisons are run against `dataset/sample_claims.csv`, "
        "which includes expected outputs for validation.\n"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline strategies")
    parser.add_argument("--strategies", nargs="+", default=["A"], choices=["A", "B", "a", "b"])
    args = parser.parse_args()

    expected = read_csv(SAMPLE_CSV)

    strategy_results = {}
    for strat in args.strategies:
        strat = strat.upper()
        print(f"\n=== Running strategy {strat} ===")
        actual, elapsed = run_strategy(strat)
        scores = score(expected, actual)
        strategy_results[strat] = {"scores": scores, "elapsed": elapsed, "n": len(actual)}

    report = format_report(strategy_results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    print(report)


if __name__ == "__main__":
    main()
