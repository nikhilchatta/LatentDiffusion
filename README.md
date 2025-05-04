# Latent Diffusion Model (LDM) — CIFAR-10 & DIV2K

This repository implements a Latent Diffusion Model (LDM) using the CompVis latent-diffusion architecture, tailored for training on CIFAR-10 and DIV2K datasets. It uses a latent autoencoder to compress image data and applies diffusion in this reduced space for efficient and high-quality image generation.

## Features

- Latent Autoencoder (AutoencoderKL / VQModel)
- Unconditional training (no transformers)
- Trains on both CIFAR-10 and DIV2K
- L2 Loss training
- Evaluation: FID, IS, LPIPS, KID, PSNR, SSIM
- Training with PyTorch Lightning and TensorBoard


## Setup Instructions

1. Clone the repository
2. Set up a Python environment (Python 3.9 recommended)
3. Install dependencies from `requirements.txt`
4. Prepare CIFAR-10 or DIV2K dataset
5. Update YAML config paths as needed

## Training

Train using the `main.py` script and a dataset-specific config. GPU support is enabled via PyTorch Lightning.

## Sampling

Generate images from the trained model using `sample.py`.

## Evaluation Metrics

Run metric scripts in the `metrics/` directory for:
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- LPIPS (Perceptual Similarity)
- KID (Kernel Inception Distance)
- PSNR and SSIM (for image restoration tasks)

## Logging

Monitor training and sampling with TensorBoard. Checkpoints and logs are saved under the `logs/` directory.

## Notes

- All training is done in latent space for memory efficiency.
- This implementation avoids transformer-based conditioning.
- Evaluation scripts expect pre-saved real and generated images.

## Credits

Based on:
- CompVis latent-diffusion
- PyTorch Lightning
- LPIPS: Learned Perceptual Image Patch Similarity



