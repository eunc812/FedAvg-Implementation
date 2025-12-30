# Federated Averaging (FedAvg) Implementation with MNIST

This repository provides a minimal PyTorch implementation of Federated Averaging (FedAvg) based on the original federated learning framework.  

The code is designed as a baseline for research, and can be easily extended to Over-the-Air Federated Learning (OTA-FL) by modifying only the aggregation module.

## Overview

Federated Learning (FL) enables multiple clients to collaboratively train a global model without sharing raw data.  
Each client performs local training on its private dataset, and the server aggregates the local updates.

## Algorithm: Federated Averaging (FedAvg)

The implementation follows **Algorithm 1** from the original FedAvg paper.
![FedAvg Algorithm](FedAvg_Algorithm.jpg)

## Training
python train.py \
  --rounds 20 \
  --clients 10 \
  --local_epochs 1 \
  --lr 0.1

## Evaluation
python eval.py --ckpt checkpoints/fedavg_mnist.pt
