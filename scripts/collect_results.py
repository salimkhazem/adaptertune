import argparse
from pathlib import Path

import json
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/summary.csv")
    args = parser.parse_args()

    rows = []
    results_dir = Path(args.results_dir)
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.yaml"
        summary_path = run_dir / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        row = {
            "run": run_dir.name,
            "dataset": cfg.get("data", {}).get("dataset"),
            "backbone": cfg.get("model", {}).get("backbone"),
            "method": cfg.get("method", {}).get("name"),
            "rank": cfg.get("method", {}).get("adapter_rank"),
            "every": cfg.get("method", {}).get("adapter_every"),
            "init": cfg.get("method", {}).get("adapter_init"),
            "seed": cfg.get("seed"),
        }
        row.update(summary)
        rows.append(row)

    if not rows:
        print("No runs found.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
