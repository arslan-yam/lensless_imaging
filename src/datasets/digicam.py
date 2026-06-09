import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lensless_helpers.preprocessor import convert_image_to_float, force_rgb, get_cropped_lensed
from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.base_dataset import BaseDataset

REPO_ID = "bezzam/DigiCam-Mirflickr-MultiMask-10K"


class DigiCamDataset(BaseDataset):
    def __init__(self, split="train", limit=None, *args, **kwargs):
        self.data = load_dataset(REPO_ID, split=split)
        labels = self.data["mask_label"]
        self.psf_cache = {label: self._simulate_psf(label) for label in sorted(set(labels))}
        index = [{"path": i, "label": int(labels[i])} for i in range(len(labels))]
        super().__init__(index, limit=limit, *args, **kwargs)

    def _simulate_psf(self, label):
        mask_path = hf_hub_download(REPO_ID, f"masks/mask_{label}.npy", repo_type="dataset")
        mask = np.load(mask_path)
        psf = simulate_psf_from_mask(mask)[0]
        return psf.permute(2, 0, 1).contiguous()

    def __getitem__(self, ind):
        entry = self._index[ind]
        item = self.data[entry["path"]]
        lensless = convert_image_to_float(force_rgb(np.array(item["lensless"])))
        lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)
        lensed = convert_image_to_float(force_rgb(np.array(item["lensed"])))
        lensed = torch.from_numpy(get_cropped_lensed(lensed, lensless))
        return {
            "lensless": lensless.permute(2, 0, 1).contiguous().float(),
            "lensed": lensed.permute(2, 0, 1).contiguous().float(),
            "psf": self.psf_cache[entry["label"]],
        }
