# Residual-Attention-Physics-Informed-Surrogate-Modeling-for-Engineering-Electro-Thermal-Fluid-Devices
# Residual-Attention Physics-Informed Surrogate Modeling for Engineering Electro-Thermal-Fluid Devices

This repository provides the training codes used in the research paper:

**Residual-Attention Physics-Informed Surrogate Modeling for Engineering Electro-Thermal-Fluid Devices**

The repository includes 16 training scripts for four benchmark cases and four PINN-based models.

## Benchmark cases

1. Case 1: Boundary-layer electro-thermal transport unit
2. Case 2: Multi-interface coupled transport channel
3. Case 3: Localized hot-spot electro-thermal transfer device
4. Case 4: Segmented electro-thermal flow-control reactor

## Models

For each benchmark case, the following models are provided:

- PINN
- LSTM-PINN
- c-LSTM-PINN
- ResAtten-PINN

## Predicted physical fields

Each model predicts five coupled physical fields:

- velocity component u
- velocity component v
- pressure p
- temperature T
- electric potential phi

## Hardware and software environment

All experiments were conducted on the same hardware platform:

- CPU: Intel Core Ultra 9 275HX
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Device: CUDA
- PyTorch: 2.2.2+cu121
- CUDA: 12.1
- Precision: torch.float32

## Installation

```bash
pip install -r requirements.txt
