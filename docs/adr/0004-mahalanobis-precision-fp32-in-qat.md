# ADR 0004: Retain FP32 Precision for Mahalanobis Distance Under INT8 QAT

* **Status:** Accepted
* **Date:** 2026-09-27
* **Author:** Goutham S

## Context and Problem Statement
To meet embedded automotive deployment constraints on edge SoCs (NVIDIA Jetson / automotive ECUs), the perception pipeline must undergo Quantization-Aware Training (QAT) to execute in INT8. However, the supervisor relies on empirical Mahalanobis distance scoring in a 32-dimensional latent space:
$$D_M(z) = \sqrt{(z - \mu_z)^T \mathbf{\Sigma}_z^{-1} (z - \mu_z)}$$
We must determine whether the Mahalanobis matrix inversion and distance computation can be quantized to INT8 alongside the neural network layers.

## Decision Drivers
* Execution latency: Must stay well inside the Fault Tolerant Time Interval (FTTI < 20 ms perception budget).
* Numerical stability: Covariance matrices can have high condition numbers; round-off error causes singular matrices.
* False Positive Rate: SOTIF monitors cannot raise spurious hazard alarms on nominal highways due to numerical noise.

## Considered Options
* Option 1: Mixed precision — INT8 QAT for dense linear layers with an FP32 bypass for Mahalanobis quadratic form
* Option 2: Full INT8 quantization for all operations including precision matrix multiplication
* Option 3: Full FP32 execution across both the network and supervisor metrics

## Decision Outcome
Chosen option: **Option 1 (Mixed Precision: INT8 Layers + FP32 Mahalanobis Bypass)**, because:
* Quantizing $\mathbf{\Sigma}_z^{-1} \in \mathbb{R}^{32 \times 32}$ to INT8 causes catastrophic cancellation and truncation error, resulting in explosive false-positive OOD flags on nominal scenes.
* Calculating the quadratic form for a 32-dimensional vector $(z - \mu)$ in FP32 takes less than **0.02 ms** on edge hardware, adding negligible latency.
* PyTorch's `DeQuantStub` allows linear layers to execute in fast INT8 while passing clean FP32 latent vectors $z$ to the distance calculator.

## Consequences
* **Positive:** Preserves the mathematical validity and low FPR of the SOTIF supervisor while capturing 75% memory bandwidth savings on the neural layers.
* **Negative:** Deployment engine (TensorRT / ONNX Runtime) requires mixed-precision configuration rather than a uniform INT8 graph.