import argparse
from pathlib import Path

import gdown

# Google Drive file ids for each trained checkpoint.
# Fill these in after uploading the trained model_best.pth files.
CHECKPOINTS = {
    "unrolled20": None,
    "modular_prepost": None,
    "modular_pre": None,
    "modular_post": None,
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
