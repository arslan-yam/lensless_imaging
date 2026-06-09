import argparse
from pathlib import Path
import torch
from lensless_helpers.preprocessor import convert_image_to_float, force_rgb, get_cropped_lensed
from src.datasets.custom_dir import IMAGE_EXTS, read_image
from src.metrics.lensless import LPIPSMetric, MSEMetric, PSNRMetric, SSIMMetric


def find_with_stem(directory, stem):
    for ext in IMAGE_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_image_chw(path):
    image = convert_image_to_float(force_rgb(read_image(path)))
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().float()


def load_lensed(lensless_path, lensed_path):
    lensless = convert_image_to_float(force_rgb(read_image(lensless_path)))
    lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)
    lensed = convert_image_to_float(force_rgb(read_image(lensed_path)))
    lensed = torch.from_numpy(get_cropped_lensed(lensed, lensless))
    return lensed.permute(2, 0, 1).contiguous().float()


def main():
    parser = argparse.ArgumentParser(description="Compute ROI metrics between reconstructions and ground truth.")
    parser.add_argument("--reconstructions", required=True, help="directory with reconstruction images")
    parser.add_argument("--dataset", required=True, help="dataset dir with lensless/ and lensed/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rec_dir = Path(args.reconstructions)
    lensless_dir = Path(args.dataset) / "lensless"
    lensed_dir = Path(args.dataset) / "lensed"
    device = args.device

    metrics = [
        PSNRMetric(name="PSNR"),
        SSIMMetric(name="SSIM"),
        MSEMetric(name="MSE"),
        LPIPSMetric(device=device, name="LPIPS"),
    ]

    totals = {m.name: 0.0 for m in metrics}
    count = 0
    for rec_path in sorted(rec_dir.iterdir()):
        if rec_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = rec_path.stem
        lensless_path = find_with_stem(lensless_dir, stem)
        lensed_path = find_with_stem(lensed_dir, stem)
        if lensless_path is None or lensed_path is None:
            continue
        
        reconstruction = load_image_chw(rec_path).unsqueeze(0).to(device)
        lensed = load_lensed(lensless_path, lensed_path).unsqueeze(0).to(device)
        batch = {"reconstruction": reconstruction, "lensed": lensed}
        for m in metrics:
            totals[m.name] += m(**batch)
        count += 1

    print(f"Evaluated {count} samples")
    
    for name, total in totals.items():
        print(f"{name:6s}: {total / count:.4f}")


if __name__ == "__main__":
    main()
