# Lensless Computational Imaging

<p align="center">
  <a href="#about">About</a> •
  <a href="#methods">Methods</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#results">Results</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template">
  <img src="https://img.shields.io/badge/built%20on-pytorch--template-green">
</a>
<img src="https://img.shields.io/badge/license-MIT-blue.svg">
</p>

## About

This repository contains training and evaluation code for reconstructing images from
[DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K) dataset.

The project is built on the [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template) (Hydra configs, comet ml logging, BaseTrainer / Inferencer).

## Methods

| Name                   | Config                     | Description                                                          |
| ---------------------- | -------------------------- | -------------------------------------------------------------------- |
| admm-100               | admm100                    | Classical admm, 100 iterations, fixed hyperparameters. No training   |
| Unrolled admm-20       | unrolled20                 | le-admm, 20 iterations, trainable per-iteration {mu1, mu2, mu3, tau} |
| modular (pre + post)   | modular_prepost            | DRUNet → Leadmm-5 → DRUNet                                           |
| modular (pre)          | modular_pre                | DRUNet → Leadmm-5                                                    |
| modular (post)         | modular_post               | Leadmm-5 → DRUNet                                                    |
| fista                  | fista / fista_unrolled     | Non-admm solver (Beck & Teboulle), fixed-100 / trainable-20          |
| admm-100 + real-esrgan | admm100_sr / admm100_sr_ft | Pretrained GAN super-resolution post-processor, frozen / fine-tuned  |

The loss is mse + lpips and all metrics (psnr, ssim, mse, lpips) are computed on the roi.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

We used rtx 5090 for training, so if you want to use rtx 50xx as well the default torch wheels have no sm_120 kernels, so reinstall
torch from the cuda 12.8 index:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

comet ml reads the api key from the environment (export it before training):

```bash
export COMET_API_KEY=your_key
```

## How To Use

### Training

The dataset downloads automatically from HuggingFace on the first run. Checkpoints are saved to saved/<run_name>/model_best.pth.

```bash
python train.py -cn=unrolled20
python train.py -cn=modular_prepost
python train.py -cn=modular_pre
python train.py -cn=modular_post
python train.py -cn=fista_unrolled
python train.py -cn=admm100_sr_ft
```

### Inference

Evaluate on the DigiCam test split (prints psnr / ssim / mse / lpips and the reconstruction speed
sec_per_image; also logs to comet ml if comet_api_key is set):

```bash
python inference.py model=modular_prepost \
    inferencer.from_pretrained=saved/modular_prepost/model_best.pth \
    datasets=digicam_eval inferencer.save_path=modular_prepost_eval
```

Parameter-free models (admm-100, fixed fista, frozen real-esrgan) take from_pretrained=null:

```bash
python inference.py model=admm100 inferencer.from_pretrained=null \
    datasets=digicam_eval inferencer.save_path=admm100_eval
```

Run on a custom directory:

```bash
python inference.py model=modular_prepost \
    inferencer.from_pretrained=saved/modular_prepost/model_best.pth \
    datasets.test.path=/path/to/data inferencer.save_path=modular_prepost
```

Reconstructions are saved as png files under `data/saved/<save_path>/test/`.

### Metrics

calculate_metrics.py compares saved reconstructions against the ground-truth lensed images on the roi:

```bash
python calculate_metrics.py \
    --reconstructions data/saved/modular_prepost/test \
    --dataset /path/to/data
```

### Demo

[`demo.ipynb`](demo.ipynb) runs the full pipeline in google colab: clone, install, download checkpoint and dataset, reconstruct → visualize → compute metrics.

### Checkpoints

Download trained checkpoints with [`scripts/download_checkpoints.py`](scripts/download_checkpoints.py).

## Credits

Built on the [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template). The lensless forward model and simulation use helpers based on [LenslessPiCam](https://github.com/LCAV/LenslessPiCam) / [waveprop](https://github.com/ebezzam/waveprop).
Method references: le-admm (Monakhova et al., 2019), DRUNet (Zhang et al., 2021), fista (Beck & Teboulle, 2009), real-esrgan (Wang et al., 2021).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
