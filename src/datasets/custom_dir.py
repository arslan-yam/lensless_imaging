import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from lensless_helpers.preprocessor import convert_image_to_float, force_rgb, get_cropped_lensed
from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.base_dataset import BaseDataset

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tiff", ".tif")


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class CustomDirDataset(BaseDataset):
    def __init__(self, path, limit=None, *args, **kwargs):
        root = Path(path)
        self.lensless_dir = root / "lensless"
        self.mask_dir = root / "masks"
        self.lensed_dir = root / "lensed"
        self.psf_cache = {}
        index = []
        
        for lensless_path in sorted(self.lensless_dir.iterdir()):
            if lensless_path.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = lensless_path.stem
            label = self._cache_psf(self.mask_dir / f"{stem}.npy")
            lensed_path = self._find(self.lensed_dir, stem)
            index.append(
                {
                    "path": str(lensless_path),
                    "label": label,
                    "name": stem,
                    "lensed_path": str(lensed_path) if lensed_path else None,
                }
            )
        super().__init__(index, limit=limit, *args, **kwargs)

    def _find(self, directory, stem):
        if not directory.exists():
            return None
        for ext in IMAGE_EXTS:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _cache_psf(self, mask_path):
        mask = np.load(mask_path)
        label = hashlib.md5(mask.tobytes()).hexdigest()
        if label not in self.psf_cache:
            psf = simulate_psf_from_mask(mask)[0]
            self.psf_cache[label] = psf.permute(2, 0, 1).contiguous()
        return label

    def __getitem__(self, ind):
        entry = self._index[ind]

        lensless = convert_image_to_float(force_rgb(read_image(entry["path"])))
        lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)

        item = {
            "lensless": lensless.permute(2, 0, 1).contiguous().float(),
            "psf": self.psf_cache[entry["label"]],
            "name": entry["name"],
        }

        if entry["lensed_path"] is not None:
            lensed = convert_image_to_float(force_rgb(read_image(entry["lensed_path"])))
            lensed = torch.from_numpy(get_cropped_lensed(lensed, lensless))
            item["lensed"] = lensed.permute(2, 0, 1).contiguous().float()

        return item
