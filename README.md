# Federated Averaging (FedAvg) Implementation with MNIST

This repository provides a minimal PyTorch implementation of Federated Averaging (FedAvg) based on the original federated learning framework.  
The code is designed as a baseline for research, and can be easily extended to Over-the-Air Federated Learning (OTA-FL) by modifying only the aggregation module.

---

## Overview

Federated Learning (FL) enables multiple clients to collaboratively train a global model **without sharing raw data**.  
Each client performs local training on its private dataset, and the server aggregates the local updates.

This implementation follows the **canonical FedAvg algorithm**:

- Clients train local models using private data
- The server aggregates local models via weighted averaging
- Communication is centralized (digital aggregation)
- MNIST is used as a reference dataset

> - This repository intentionally keeps the model and pipeline simple to facilitate **research extensions**, such as:
> - Over-the-Air Computation (AirComp)
> - Analog aggregation
> - Device selection and beamforming optimization

---

## Algorithm: Federated Averaging (FedAvg)

The implementation follows **Algorithm 1** from the original FedAvg paper.

![FedAvg Algorithm](FedAvg_Algorithm.jpg)


---

## Repository Structure

