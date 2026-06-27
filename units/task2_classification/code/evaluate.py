from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stored Task2 run metrics.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    metrics_path = Path(args.run_dir) / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    print(json.dumps(json.loads(metrics_path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
