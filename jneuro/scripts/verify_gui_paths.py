import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def measure(catalog: Path, reports: Path) -> dict:
    paths = {}
    for line in catalog.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(GUI-\d+)\s*\|\s*(.*?)\s*\|\s*`(\w+)`\s*\|", line)
        if match:
            identifier, description, method = match.groups()
            if identifier in paths:
                raise ValueError(f"Duplicate GUI path: {identifier}")
            paths[identifier] = {"description": description, "method": method}
    if not paths:
        raise ValueError("GUI path inventory is empty.")
    outcomes = {}
    for report in reports.glob("*.xml"):
        for case in ET.parse(report).getroot().iter("testcase"):
            if case.get("classname") != "com.lis.neuro.NeuroGuiTest":
                continue
            method = case.get("name", "").split("(")[0]
            successful = not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
            outcomes[method] = outcomes.get(method, True) and successful
    covered = [identifier for identifier, path in paths.items() if outcomes.get(path["method"], False)]
    missing = [identifier for identifier in paths if identifier not in covered]
    return {"total": len(paths), "covered": len(covered), "ratio": len(covered) / len(paths),
            "covered_paths": covered, "uncovered_paths": missing, "paths": paths, "test_outcomes": outcomes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.catalog, args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"GUI path coverage: {result['covered']}/{result['total']} = {result['ratio']:.2%}; required >=90%")
    if result["ratio"] < 0.90:
        print("Uncovered: " + ", ".join(result["uncovered_paths"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
