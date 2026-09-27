# ADR 0002: Use Frozen ResNet-50 for Feature Extraction Over DINOv2

* **Status:** Accepted
* **Date:** 2026-09-27
* **Author:** Goutham S

## Context & Problem Statement
The safety supervisor requires high-level semantic scene context to evaluate out-of-distribution (OOD) risks. We evaluated whether to use a modern Vision Transformer (DINOv2 ViT-L/14) or a classical CNN (ResNet-50).

## Decision Drivers
* Strict automotive Fault Tolerant Time Interval (FTTI) limit (< 20 ms budget for perception).
* VRAM and compute constraints for local training on GTX 1650 (4 GB).
* Requirement for a well-conditioned multivariate Gaussian latent space for Mahalanobis distance scoring.

## Considered Options
1. DINOv2 (ViT-L/14)
2. ResNet-50 (ImageNet-1K pretrained)
3. YOLOv8n internal backbone tapping (SPPF layer)

## Decision Outcome
Chosen option: **Option 2 (ResNet-50)**, because:
* Latency on edge SoCs is ~6 ms in INT8/FP16 (versus >90 ms for DINOv2).
* Global average pooling produces a compact 2048-d vector that forms clean Gaussian clusters under empirical covariance modeling.
* Independent from the primary YOLO detector, preventing common-cause failures (ISO 26262 / ISO 21448).

## Consequences
* **Positive:** Extraction runs stably on local hardware; feature files are small (~330 MB); high throughput.
* **Negative:** ResNet-50 lacks fine-grained patch-level attention tokens that modern foundation models offer.