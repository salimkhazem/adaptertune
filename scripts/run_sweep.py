import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="Base configs list")
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--every", default="1")
    parser.add_argument("--init", default="zero")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="0")
    parser.add_argument("--exp-name", default="rank_sweep")
    args = parser.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]
    every = int(args.every)
    seeds = [int(s) for s in args.seeds.split(",")]

    for seed in seeds:
        for rank in ranks:
            override = [
                f"seed={seed}",
                f"method.adapter_rank={rank}",
                f"method.adapter_every={every}",
                f"method.adapter_init={args.init}",
                f"logging.exp_name={args.exp_name}_r{rank}_s{seed}",
            ]
            cmd = [sys.executable, "-m", "scripts.train"]
            for cfg in args.configs:
                cmd += ["--config", cfg]
            for ov in override:
                cmd += ["--override", ov]
            cmd += ["--device", args.device]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
