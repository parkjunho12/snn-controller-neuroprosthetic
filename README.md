# 🧠 Designing Spiking Neural Controllers for Neuroprosthetic Systems

> MSc Dissertation — The University of Sheffield  
> **Author:** Junho Park  
> **Supervisor:** Dr. Luca Manneschi  
> **Department of Computer Science**  
> **Date:** September 2025  

---

## 📘 Overview

This repository accompanies the MSc dissertation titled  
**“Designing Spiking Neural Controllers for Neuroprosthetic Systems.”**

The research investigates spiking-based neural architectures for surface electromyography (sEMG)–driven neuroprosthetic control.  
Five model families were benchmarked on the **NinaPro DB6** dataset:

| Model Type | Description |
|-------------|-------------|
| **LSTM** | Baseline sequential model for temporal dependency learning |
| **TCN-only** | Dilated causal convolution for long-range dependency modeling |
| **SNN-only** | Pure event-driven Leaky Integrate-and-Fire (LIF) spiking model |
| **SpikingTCN** | TCN blocks integrated with spiking neurons |
| **Hybrid TCN–SNN** | Parallel feature fusion of TCN and SNN modules |

---

## 🎯 Research Objectives

- Develop an **energy-efficient neural controller** for sEMG-based gesture recognition.  
- Explore **spike encoding schemes** — Rate, Delta, and Latency — and their trade-offs.  
- Benchmark **five neural architectures** for accuracy, energy efficiency, and real-time feasibility.  
- Analyze **firing rate dynamics**, spike sparsity, and inference energy as proxies for neuromorphic efficiency.

---

## ⚙️ Experimental Setup

**Dataset:** [NinaPro DB6](https://ninapro.hevs.ch/)  
**Input:** 14-channel sEMG, 2 kHz sampling rate  
**Windowing:** 200-sample sliding window (50 % overlap)  
**Spike Encodings:** `rate`, `delta`, `latency`  
**Timesteps:** Ts = 20  
**Batch size:** 32  
**Optimizer:** Adam (with early stopping)  
**Metrics:** Accuracy, Macro-F1, spike counts, mean firing rate

<p align="center">
  <img src="docs/framework.png" width="700"/>
  <br/>
  <em>Overall workflow: preprocessing → encoding → model → evaluation.</em>
</p>

---

## 🧩 Model Architectures

### 1. Temporal Convolutional Network (TCN)
- Causal & dilated convolutions  
- Residual connections for stable gradient flow  
- Parallelizable sequence modeling  

### 2. Spiking Neural Network (SNN)
- Leaky Integrate-and-Fire neurons  
- Surrogate-gradient learning (ATan / fast-sigmoid)  
- Event-driven computation for low power inference  

### 3. Hybrid TCN–SNN
- Dual-branch fusion: TCN feature extractor + SNN firing-rate encoder  
- Achieves balance between accuracy and sparsity  
- Designed for real-time neuroprosthetic control  

---

## 📊 Key Results

| Model | Encoding | Accuracy (%) | Macro-F1 (%) | Avg. Firing Rate |
|:------|:----------|:--------------|:--------------|:-----------------|
| **LSTM** | Rate | 82.1 | 80.5 | – |
| **TCN-only** | Rate | **85.0** | **84.7** | – |
| **SNN-only** | Delta | 62.4 | 61.9 | 5–20 % |
| **SpikingTCN** | Rate | 76.6 | 75.9 | 10–18 % |
| **Hybrid TCN–SNN** | Delta | **88.0** | **87.8** | **≈ 3 %** |

> 🔋 The **Hybrid TCN–SNN** achieved the **best trade-off** between accuracy and energy efficiency, reducing total spike activity by >10× compared to SNN-only.

<p align="center">
  <img src="docs/accuracy_vs_energy.png" width="550"/>
  <br/>
  <em>Accuracy–Energy trade-off across architectures.</em>
</p>

---

## 🧠 Methodology

1. **Preprocessing**
   - Band-pass filtering and z-score normalization per channel  
   - Sliding window segmentation (200 samples / 100 overlap)  
   - Label majority voting for gesture stability  

2. **Encoding**
   - `rate`: probability-based Bernoulli firing  
   - `latency`: timing-based first-spike encoding  
   - `delta`: event-based threshold triggering  

3. **Training**
   - PyTorch + snntorch pipeline  
   - Surrogate gradient backpropagation through time (BPTT)  
   - Cross-entropy loss over firing-rate logits  

4. **Evaluation**
   - Accuracy, F1-score, confusion matrices  
   - Spike-based energy metrics (mean spike count, synaptic events)

---

## 🧪 Environment Setup

```bash
# Clone repository
git clone https://github.com/parkjunho12/edge-snn-robot-template.git
cd edge-snn-robot-template

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run experiment
python src/train_hybrid.py --dataset ninapro_db6 --encoding delta
