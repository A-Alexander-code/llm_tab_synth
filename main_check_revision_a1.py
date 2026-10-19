"""Load all A1 arrays and exercise isolated consumers without any training."""
import argparse
from pathlib import Path

from src.revision_a1 import write_json
from src.revision_a1_runtime import RevisedRun
from src.revision_a1_audit import audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-a1-run", required=True)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory; never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(RevisedRun(args.revision_a1_run))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", result)
    print(f"Verified {result['verified_combinations']} combinations; no training. Evidence: {args.output / 'report.json'}")
    for name, info in result["split_diversity"].items():
        print(f"{name}: {info['distinct_test_sets']} distinct test sets")
