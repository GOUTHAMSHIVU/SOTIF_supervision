# ADR 0003: Cache Penultimate ResNet-50 Feature Embeddings in HDF5 (FP32)

* **Status:** Accepted
* **Date:** 2026-09-27
* **Author:** Goutham S

## Context and Problem Statement
Training Head A (Beta-VAE) and Head B (Shift Regressor) directly on raw $1920 \times 1208$ A2D2 PNG frames requires repeated disk I/O, CPU decompression, and redundant forward passes through a frozen ResNet-50 backbone. On a local GTX 1650 (4 GB VRAM) or cloud instance, this creates severe training bottlenecks and unnecessary compute costs. How should intermediate representations be stored?

## Decision Drivers
* Training throughput: Supervisor heads require rapid epoch iterations (50+ epochs).
* Disk storage and I/O efficiency: Raw images take ~95 GB; training reads must not saturate PCIe/SATA bandwidth.
* Precision preservation: Reconstruction loss and covariance matrix estimation require uncorrupted numerical fidelity.

## Considered Options
* Option 1: Offline single-pass extraction to HDF5 (`.h5`) container in standard FP32
* Option 2: End-to-end on-the-fly image streaming from disk through ResNet-50 per epoch
* Option 3: Save intermediate embeddings as individual PyTorch `.pt` tensors per image
* Option 4: Quantize embeddings to INT8 on disk during extraction

## Decision Outcome
Chosen option: **Option 1 (Offline HDF5 Feature Caching in FP32)**, because:
* **Footprint Reduction:** Compresses ~95 GB of raw images into a single contiguous ~338 MB `.h5` file ($41{,}277 \times 2048 \times 4$ bytes).
* **RAM Residency:** The entire dataset can be loaded into system RAM, eliminating disk access during VAE training.
* **Epoch Speedup:** Head A epochs drop from ~25 minutes to under 20 seconds.
* **Numerical Integrity:** Retaining FP32 prevents quantization noise from biasing the baseline reconstruction target or skewing covariance matrices.

## Consequences
* **Positive:** Massive reduction in training runtime; zero redundant backbone GPU computations; reproducible static dataset.
* **Negative:** High one-time extraction latency (~15–17 hours on GTX 1650); separate cached datasets required for corrupted imagery (Head B).