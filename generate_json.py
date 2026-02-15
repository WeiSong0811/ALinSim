#!/usr/bin/env python3
"""
Generate a search-space JSON file for run_all_methods.py (search-space mode).

Edit FEATURE_SPECS below, then run:
    python generate_search_space_json.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


# Output JSON path (relative to repository root)
OUTPUT_JSON = Path("data/search_space.json")

# Edit your features here.
# Required keys per feature:
# - name: feature name
# - min: lower bound
# - max: upper bound
# - count: number of values in [min, max]
# Optional:
# - type: "int" or "float" (default: "float")
FEATURE_SPECS: List[Dict[str, Any]] = [
    {"name": "PS:PAN ratio", "min": 0.1, "max": 0.99, "count": 10, "type": "float"},
    {"name": "Feed rate(mL/h)", "min": 0.1, "max": 0.99, "count": 10, "type": "float"},
    {"name": "Distance(cm)", "min": 0.1, "max": 0.99, "count": 10, "type": "float"},
    {"name": "Mass fraction of solute", "min": 0.1, "max": 0.99, "count": 10, "type": "float"},
    {"name": "Mass fraction of SiO2 in solute ", "min": 0.1, "max": 0.99, "count": 10, "type": "float"},
    {"name": "Applied voltage(kV)", "min": 0.1, "max": 0.99, "count": 10, "type": "float"},
    {"name": "Inner diameter(mm)", "min": 0.1, "max": 0.99, "count": 10, "type": "float"},
]
def validate_feature_spec(spec: Dict[str, Any]) -> None:
    required = {"name", "min", "max", "count"}
    missing = required - set(spec.keys())
    if missing:
        raise ValueError(f"Feature spec missing keys {missing}: {spec}")

    name = spec["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Invalid feature name: {name}")

    low = spec["min"]
    high = spec["max"]
    if low > high:
        raise ValueError(f"Feature '{name}': min ({low}) > max ({high})")

    count = int(spec["count"])
    if count <= 0:
        raise ValueError(f"Feature '{name}': count must be > 0")

    dtype = str(spec.get("type", "float")).lower()
    if dtype not in {"int", "float"}:
        raise ValueError(f"Feature '{name}': type must be 'int' or 'float'")


def build_search_space(feature_specs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    search_space: Dict[str, Dict[str, Any]] = {}
    names = set()

    for raw_spec in feature_specs:
        spec = dict(raw_spec)
        validate_feature_spec(spec)
        name = spec["name"].strip()
        if name in names:
            raise ValueError(f"Duplicate feature name: {name}")
        names.add(name)

        search_space[name] = {
            "min": spec["min"],
            "max": spec["max"],
            "count": int(spec["count"]),
            "type": str(spec.get("type", "float")).lower(),
        }

    if not search_space:
        raise ValueError("FEATURE_SPECS is empty.")
    return search_space


def main() -> None:
    search_space = build_search_space(FEATURE_SPECS)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(search_space, f, indent=2, ensure_ascii=False)
    print(f"Search-space JSON written to: {OUTPUT_JSON.resolve()}")
    print(f"Features: {', '.join(search_space.keys())}")


if __name__ == "__main__":
    main()
