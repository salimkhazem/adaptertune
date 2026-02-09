import argparse
import itertools
import subprocess
import sys


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0", help="Comma-separated CUDA device ids")
    args = parser.parse_args()

    devices = args.devices.split(",")
    dev_cycle = itertools.cycle(devices)

    base = "configs/base.yaml"
    cifar10 = "configs/datasets/cifar10.yaml"
    vit_small = "configs/models/vit_small.yaml"
    quick = "configs/experiments/quick_sanity.yaml"
    rank_quick = "configs/experiments/rank_sweep_quick.yaml"

    # 1) fast sanity: head_only and adapter_tune
    for method in ["head_only", "adapter_tune"]:
        device = next(dev_cycle)
        run([
            sys.executable, "-m", "scripts.train",
            "--config", base,
            "--config", cifar10,
            "--config", vit_small,
            "--config", quick,
            "--override", f"method.name={method}",
            "--override", f"logging.exp_name=sanity_{method}",
            "--device", device,
        ])

    # 2) rank sweep quick
    device = next(dev_cycle)
    run([
        sys.executable, "-m", "scripts.run_sweep",
        "--configs", base, cifar10, vit_small, rank_quick,
        "--ranks", "8,16,32,64",
        "--seeds", "0,1,2",
        "--device", device,
        "--exp-name", "rank_quick",
    ])

    # 3) toy experiment
    run([sys.executable, "-m", "scripts.toy_2d"])

    # 4) collect + tables + figures
    run([sys.executable, "-m", "scripts.collect_results"])
    run([sys.executable, "-m", "scripts.make_tables"])
    run([sys.executable, "-m", "scripts.make_figures"])


if __name__ == "__main__":
    main()
