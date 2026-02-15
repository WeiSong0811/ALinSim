#!/usr/bin/env python3
"""
Export round-1 selected candidates from search_space_results.json to CSV.

Default behavior:
- Read all methods under output/*
- Extract rounds[0].selected_candidates
- Write one CSV per method to output/round1_exports/
- Include target columns as empty placeholders for real labels
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# Edit here
OUTPUT_ROOT = Path("output")
EXPORT_DIR = OUTPUT_ROOT / "exports"
METHODS: List[str] = []  # [] means auto-discover all methods with search_space_results.json


def discover_result_files(output_root: Path, methods: List[str]) -> List[Path]:
    if methods:
        files = [output_root / m / "search_space_results.json" for m in methods]
        return [p for p in files if p.exists()]
    return sorted(output_root.glob("*/search_space_results.json"))


def to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    method = payload.get("method", "unknown_method")
    target_columns = payload.get("target_columns", [])
    rounds = payload.get("rounds", [])
    if not rounds:
        raise ValueError("No rounds found in result JSON.")

    round1 = rounds[0]
    candidates = round1.get("selected_candidates", [])
    if not candidates:
        raise ValueError("Round 1 has no selected_candidates.")

    df = pd.DataFrame(candidates)
    # Keep metadata so each row can be traced.
    df.insert(0, "method", method)
    # Add label columns as empty placeholders for real experiment outcomes.
    for col in target_columns:
        if col not in df.columns:
            df[col] = pd.NA

    return df
def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = discover_result_files(OUTPUT_ROOT, METHODS)
    if not files:
        raise FileNotFoundError("No search_space_results.json files found.")

    exported = []
    failed = []
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            df = to_dataframe(payload)
            method = payload.get("method", file_path.parent.name)
            out_file = EXPORT_DIR / f"{method}_candidates.csv"
            df.to_csv(out_file, index=False, encoding="utf-8")
            exported.append(str(out_file))
        except Exception as e:
            failed.append({"file": str(file_path), "reason": str(e)})

    summary = {
        "exported_count": len(exported),
        "failed_count": len(failed),
        "exported_files": exported,
        "failed": failed,
    }
    summary_file = EXPORT_DIR / "export_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Done. Exported: {len(exported)}, Failed: {len(failed)}")
    print(f"CSV folder: {EXPORT_DIR.resolve()}")
    print(f"Summary: {summary_file.resolve()}")


if __name__ == "__main__":
    main()
