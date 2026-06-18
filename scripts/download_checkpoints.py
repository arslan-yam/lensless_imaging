import argparse
from pathlib import Path
import gdown

CHECKPOINTS = {
    "unrolled20": "1YlikM_FETV_s8LwT8v-zV--dZqCR2lcL",
    "modular_prepost": "18i4-3Sb5zqrdKVhE72NMyvIhOgSxwsY2",
    "modular_pre": "1aEeT9QZOjuqOA4LXeAH2_LT8VExJ7QqL",
    "modular_post": "199p_SAbgD0-UgQ66oVO9kVpzl4498U1k",
    "fista_unrolled": "1L8EYTaAvKuOw3QB1WN0TOIhP52KrFSpi",
    "admm_100_sr_ft": "1c1VBP2zzZ5F3V1vSgs5vetIRCJJJckvU"
}


def main():
    parser = argparse.ArgumentParser(description="Download trained checkpoints.")
    parser.add_argument("--out", default="saved", help="output directory")
    args = parser.parse_args()

    out = Path(args.out)
    for name, file_id in CHECKPOINTS.items():
        if file_id is None:
            print(f"Skipping {name}: no file id set")
            continue
        dst = out / name / "model_best.pth"
        dst.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=file_id, output=str(dst), quiet=False)


if __name__ == "__main__":
    main()
