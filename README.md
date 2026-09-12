<h1 align="center">Lightweight Multimodal Person Detection and Temporal Post-Processing for Aerial Search and Rescue</h1>
<p align="center"><strong>(MMSAR)</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Resolution-640%C3%97640-555?style=flat" alt="Resolution: 640×640">
  <img src="https://img.shields.io/badge/Sampling_Rate-8_Hz_(125_ms)-blue?style=flat" alt="Sampling Rate: 8 Hz">
  <img src="https://img.shields.io/badge/Backbones-2%C3%97_YOLO11n_(2.6M_Params)-orange?style=flat" alt="Backbone: YOLO11n">
  <img src="https://img.shields.io/badge/Status-ICSPIS_2026_Submission-red?style=flat" alt="Status: ICSPIS 2026 Submission">
</p>

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Research Framework](#research-framework)
  - [Research Question](#research-question)
  - [Core Contributions](#core-contributions)
  - [Upstream Detection Backbones](#upstream-detection-backbones)
  - [Multimodal Late Decision Fusion](#multimodal-late-decision-fusion)
  - [Causal Temporal Post-Processing Methods](#causal-temporal-post-processing-methods)
  - [Theoretical Complexity & State Memory](#theoretical-complexity--state-memory)
  - [Validation Threshold Optimization & Bootstrapping](#validation-threshold-optimization--bootstrapping)
  - [Downstream Operational Resource Impact](#downstream-operational-resource-impact)
- [Dataset Corpus & Harvesting](#dataset-corpus--harvesting)
  - [Controlled 2×2 Experimental Design](#controlled-22-experimental-design)
  - [Dual-Crop Harvesting Procedure](#dual-crop-harvesting-procedure)
  - [Decimation & Sequence-Level 60/20/20 Split](#decimation--sequence-level-602020-split)
  - [Dataset Composition Matrix](#dataset-composition-matrix)
- [Results & Benchmarks](#results--benchmarks)
  - [Table 1: Upstream Frame-Level Detection Performance](#table-1-upstream-frame-level-detection-performance)
  - [Table 2: Comparative Benchmark of Causal Post-Processing](#table-2-comparative-benchmark-of-causal-post-processing)
  - [Table 3: Environment-Stratified Operational Resource Impact](#table-3-environment-stratified-operational-resource-impact)
- [Quick Reproduction](#quick-reproduction)
- [Repository Organization](#repository-organization)
- [Authors & Citation](#authors--citation)
- [Acknowledgments & License](#acknowledgments--license)

---

## Overview

Autonomous unmanned aerial vehicles (UAVs) deployed in wilderness search-and-rescue (SAR) missions require dependable person detection under severe size, weight, power, and computing (SWaP-C) constraints. While multimodal sensing combining visible red-green-blue (RGB) and long-wave thermal infrared (TIR) sensors mitigates single-modality dropouts caused by canopy shade or thermal crossover, frame-level deep learning detectors produce transient false positive spikes. In autonomous aerial operations, each false alarm triggers costly loiter maneuvers, battery depletion, and spurious emergency telemetry transmissions over bandwidth-constrained satellite or cellular Internet of Things (IoT) uplinks.

This repository contains the official implementation, dataset harvesting pipeline, and comparative evaluation framework for the paper:

> **"Lightweight Multimodal Person Detection and Temporal Post-Processing for Aerial Search and Rescue"**  
> *Oumar Mamoun Ibrahim and Mohamad Khairi bin Ishak*  
> Planned for submission to the 10th International Conference on Signal Processing and Integrated Networks (**ICSPIS 2026**).

---

## Pipeline Architecture

```mermaid
flowchart TD
    subgraph S1 ["Stage 1: Video Capture & Preprocessing"]
        RGB_raw["Raw RGB Video\n1280x720 @ 24 FPS"]
        TIR_raw["Raw Thermal Video\n1280x720 @ 24 FPS"]
        Crop_RGB["Dual-Crop Extraction\nLeft (x=0) & Right (x=640)\n640x640 Snippets (y=40)"]
        Crop_TIR["Dual-Crop Extraction\nLeft (x=0) & Right (x=640)\n640x640 Snippets (y=40)"]
        Dec_RGB["3:1 Decimation\nTimebase fs = 8 Hz (Ts = 125 ms)"]
        Dec_TIR["3:1 Decimation\nTimebase fs = 8 Hz (Ts = 125 ms)"]
    end

    subgraph S2 ["Stage 2: Parallel Upstream Detectors"]
        YOLO_RGB["YOLO11n-RGB Backbone\n2.6M params | 6.5 GFLOPs\nFP32 Precision | Batch 16"]
        YOLO_TIR["YOLO11n-Thermal Backbone\n2.6M params | 6.5 GFLOPs\nFP32 Precision | Batch 16"]
    end

    subgraph S3 ["Stage 3: Decision Fusion Gate"]
        LateFusion["Soft Disjunctive Late Fusion\ns[n] = max(c_RGB[n], c_Thermal[n])\nScalar Confidence s[n] in [0, 1]"]
    end

    subgraph S4 ["Stage 4: Causal Temporal Post-Processing (W = 5 frames / 625 ms)"]
        direction TB
        M1["M1: Baseline Raw Thresholding\ny_raw[n] = I(s[n] >= tau_raw*)"]
        M2["M2: 5-Frame Moving Average\ny_MA[n] = I(s_MA[n] >= tau_MA*)"]
        M3["M3: 5-Frame Median Filter\ny_med[n] = I(s_med[n] >= tau_med*)"]
        M4["M4: 5-Frame History Consensus\ny_hist[n] = I(Sum I(s[n-k] >= tau_hist*) >= 3)"]
        M5["M5: Learned Mamba-SSSM\ny_ssm[n] = I(sigma(W_out*u_t + b_out) >= tau_ssm*)\n753 params | 512 B state memory"]
    end

    subgraph S5 ["Stage 5: Mission Decision & Operational Impact"]
        Alarm["Binary Alarm Decision\ny[n] in {0, 1}"]
        Telemetry["Emergency IoT Telemetry\nS_pkt = 1.2 kB (pkt) / 45 kB (thumb)"]
        Loiter["UAV Loiter Maneuver\nP_hover = 280 W | T_loiter in [15, 30] s\nE_loiter = 4.2 - 8.4 kJ per event"]
    end

    RGB_raw --> Crop_RGB --> Dec_RGB --> YOLO_RGB
    TIR_raw --> Crop_TIR --> Dec_TIR --> YOLO_TIR

    YOLO_RGB -->|"c_RGB[n] in [0.0, 1.0]"| LateFusion
    YOLO_TIR -->|"c_Thermal[n] in [0.0, 1.0]"| LateFusion

    LateFusion -->|"Observation Window W[n] = {s[n-4], ..., s[n]}"| M1 & M2 & M3 & M4 & M5

    M1 & M2 & M3 & M4 & M5 --> Alarm

    Alarm -->|"y[n] = 1 (Alarm Asserted)"| Telemetry
    Alarm -->|"y[n] = 1 (Target Verification)"| Loiter
```

---

## Research Framework

### Research Question
> *How do causal temporal post-processing methods compare in reducing false alarms across visual modalities (RGB versus thermal infrared) and wilderness environments (arid desert versus temperate forest) in lightweight aerial detection?*

### Core Contributions
1. **Controlled 2×2 Video Corpus:** A dual-stream RGB and long-wave thermal infrared (TIR) benchmark across arid desert and temperate forest environments, systematically structured into positive target scenarios, challenging hard-negative distractors (sun-heated rocks, animal clutter, moving canopy shadows), and clear-negative scenes.
2. **Modality-Specific Detection Baselines:** A standardized edge detection framework deploying two separate Ultralytics YOLO11n models (2.6M parameters, 6.5 GFLOPs at $640 \times 640$): one specialized on multi-environment RGB sequences and one specialized on multi-environment thermal sequences, trained in single-precision floating-point (FP32) arithmetic with batch size 16 for deterministic numerical stability on embedded avionics.
3. **Comparative Post-Processing Benchmark:** An empirical evaluation of five causal post-processing techniques (raw thresholding, 5-frame moving average, 5-frame median filtering, 5-frame history consensus, and a pure-PyTorch Mamba selective state-space model), with each method operating under a validation-optimized decision threshold $\tau_m^*$.
4. **Modality and Environment Sensitivity Analysis:** Systematic characterization of how causal temporal filters suppress false alarms differently across RGB and thermal confidence streams, and across disparate wilderness biomes (thermal crossover in desert vs. canopy shadow occlusion in forest).
5. **Operational Resource Impact Analysis:** Rigorous modeling of downstream benefits, translating false alarm suppression into preserved UAV battery reserves, expanded search flight endurance, and conserved satellite/cellular IoT telemetry bandwidth.

---

### Upstream Detection Backbones

To isolate temporal post-processing dynamics without conflating single-model weights across disparate spectral representations, we deploy two dedicated unimodal detectors:
- **YOLO11n-RGB:** Specialized on visible RGB imagery ($c_{\text{RGB}}[n] \in [0.0, 1.0]$).
- **YOLO11n-Thermal:** Specialized on thermal infrared imagery ($c_{\text{Thermal}}[n] \in [0.0, 1.0]$).

<div align="center">

| Model Backbone | Modality | Input Resolution | Parameters | Computational Cost | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n-RGB** | Visible RGB | $640 \times 640$ px | 2.6M | 6.5 GFLOPs | Full FP32 (`amp=False`) |
| **YOLO11n-Thermal** | Thermal TIR | $640 \times 640$ px | 2.6M | 6.5 GFLOPs | Full FP32 (`amp=False`) |

</div>

#### Detector Training Configuration

<div align="center">

| Training Parameter | Configuration Value | Rationale / Specification |
| :--- | :--- | :--- |
| **Batch Size** | 16 | Optimal gradient variance for small target feature representation |
| **Arithmetic Precision** | Full FP32 (`amp=False`) | Prevents underflow and numerical instability on edge embedded GPUs |
| **Epochs** | 100 | Complete convergence across multi-environment sequence splits |
| **Early Stopping** | Disabled (`patience=0`) | Preserves uniform training trajectory across both modality branches |
| **Optimizer** | SGD | Momentum: 0.937, Weight Decay: 0.0005 |
| **Learning Rate** | $\text{lr}_0 = 0.01$ | Cosine decay schedule to final $\text{lr}_f = 0.0001$ |
| **Hardware GPU** | NVIDIA GeForce RTX 4060 | 8 GB Dedicated VRAM |

</div>

#### Real-Time Edge Budget Constraint
At an operating timebase of $f_s = 8\text{ Hz}$ ($T_s = 125\text{ ms}$), the pipeline satisfies frame-synchronous execution without buffering delays:
- **Serial Execution:** $T_{\text{RGB}} + T_{\text{Thermal}} + T_{\text{fusion}} + T_{\text{postproc}} \le 125\text{ ms}$
- **Parallel Multi-Core Execution:** $\max(T_{\text{RGB}}, \, T_{\text{Thermal}}) + T_{\text{fusion}} + T_{\text{postproc}} \le 125\text{ ms}$

---

### Multimodal Late Decision Fusion

To preserve true human detections when one sensor branch is compromised without multiplying miss rates, detector confidence scores are integrated via soft disjunctive (OR) late fusion via max-pooling:

$$s[n] = \max\big(c_{\text{RGB}}[n], \, c_{\text{Thermal}}[n]\big) \in [0.0, 1.0]$$

This scalar fused confidence $s[n]$ feeds directly into downstream causal temporal post-processing.

---

### Causal Temporal Post-Processing Methods

All methods operate causally over a sliding window of length $W = 5$ frames ($625\text{ ms}$ latency budget at $8\text{ Hz}$):
$$\mathcal{W}[n] = \big\{s[n-4], \, s[n-3], \, s[n-2], \, s[n-1], \, s[n]\big\}$$

1. **M1 — Baseline Raw Thresholding (Instantaneous):**
   $$y_{\text{raw}}[n] = \mathbb{I}(s[n] \ge \tau_{\text{raw}}^*)$$
   Memoryless per-frame baseline. Vulnerable to isolated single-frame distractor spikes.

2. **M2 — Five-Frame Moving Average (Sliding Mean):**
   $$\tilde{s}_{\text{MA}}[n] = \frac{1}{W} \sum_{k=0}^{W-1} s[n-k], \qquad y_{\text{MA}}[n] = \mathbb{I}\big(\tilde{s}_{\text{MA}}[n] \ge \tau_{\text{MA}}^*\big)$$
   Attenuates an isolated single-frame spike of $1.0$ down to $0.20$. Lowers $\tau_{\text{MA}}^*$ to maintain target onset sensitivity.

3. **M3 — Five-Frame Median Filter (Order-Statistic):**
   $$\tilde{s}_{\text{med}}[n] = \text{median}\big(s[n], \, s[n-1], \, \dots, \, s[n-4]\big), \qquad y_{\text{med}}[n] = \mathbb{I}\big(\tilde{s}_{\text{med}}[n] \ge \tau_{\text{med}}^*\big)$$
   Completely suppresses impulse noise bursts spanning fewer than $\lfloor W/2 \rfloor + 1 = 3$ frames while preserving step edges.

4. **M4 — Five-Frame History Consensus ($M$-out-of-$N$ Voting):**
   $$y_{\text{hist}}[n] = \mathbb{I}\left( \sum_{k=0}^{W-1} \mathbb{I}\big(s[n-k] \ge \tau_{\text{hist}}^*\big) \ge 3 \right)$$
   Discrete binary consensus requiring at least 3 out of 5 consecutive frames to independently exceed candidate threshold $\tau_{\text{hist}}^*$.

5. **M5 — Learned Mamba Selective State-Space Model (Mamba-SSSM):**
   Pure-PyTorch implementation of a selective state-space sequence model ($d_{\text{model}} = 16, d_{\text{state}} = 8$, 753 parameters):
   $$\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta), \quad \bar{A}_t = \exp(\Delta_t \cdot A), \quad \bar{B}_t = (\Delta_t \cdot B_t) \odot x_t$$
   $$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t, \quad u_t = \sum_{j=1}^{d_{\text{state}}} h_{t, j} \odot C_{t, j}$$
   $$y_{\text{ssm}}[n] = \mathbb{I}\big(\sigma(W_{\text{out}} u_t + b_{\text{out}}) \ge \tau_{\text{ssm}}^*\big)$$
   Maintains recurrent hidden state $h_t$ in exactly 512 bytes of memory, achieving strictly causal $O(1)$ per-frame execution.

---

### Theoretical Complexity & State Memory

<div align="center">

| Post-Processing Method | Computational Complexity | Trainable Parameters | State Memory Footprint |
| :--- | :---: | :---: | :---: |
| **M1: Raw Thresholding** | $O(1)$ | 0 | 0 Bytes (Memoryless) |
| **M2: Moving Average** | $O(1)$ amortized$^*$ | 0 | $5 \times 4\text{ B} = 20\text{ Bytes}$ |
| **M3: Median Filter** | $O(W)$ (Sorted Insert) | 0 | $5 \times 4\text{ B} = 20\text{ Bytes}$ |
| **M4: History Consensus** | $O(W)$ | 0 | $5\text{ Bytes}$ (Packed Booleans) |
| **M5: Mamba-SSSM** | $O(d_{\text{model}} \cdot d_{\text{state}})$ | 753 | $16 \times 8 \times 4\text{ B} = 512\text{ Bytes}$ |

</div>

$^*$*With circular buffer / running sum.*

---

### Validation Threshold Optimization & Bootstrapping

- **Threshold Optimization ($\tau_m^*$):**  
  Enforcing a uniform arbitrary threshold across filters with distinct transfer functions penalizes performance. For each method $m$, the optimal decision threshold $\tau_m^*$ is farmed exclusively on the disjoint validation partition $\mathcal{D}_{\text{val}}$:
  $$\tau_m^* = \arg\max_{\tau \in [0.05, 0.95]} F_1\big(\tau; \, \mathcal{D}_{\text{val}}, \, \mathcal{M}_m\big)$$
  sweeping with step $\Delta \tau = 0.02$ (46 evaluation points). Once identified on validation sequences, thresholds are permanently frozen prior to test split evaluation.
- **Sequence-Level Bootstrapping:**  
  Because sequential video frames exhibit temporal correlation, standard parametric normality assumptions fail. We resample the 12 held-out test clips with replacement across $B = 1000$ bootstrap iterations, computing empirical two-sided 95% percentile confidence intervals ($2.5\text{th}$ and $97.5\text{th}$ percentiles) for F1-score.

---

### Downstream Operational Resource Impact

In autonomous aerial SAR, false alarms incur mission-critical overhead:
- **False Alarm Suppression Ratio (FASR):**
  $$\text{FASR} = 1 - \frac{\sum_n \mathbb{I}(y_{\text{method}}[n] = 1 \wedge y_{\text{true}}[n] = 0)}{\sum_n \mathbb{I}(y_{\text{baseline}}[n] = 1 \wedge y_{\text{true}}[n] = 0)}$$
- **Uplink Telemetry Bandwidth Preserved:**  
  Confirmed alarms transmit packets over low-bandwidth satellite/cellular links ($S_{\text{pkt}} = 1.2\text{ kB}$ standard telemetry, $45\text{ kB}$ with thumbnail):
  $$\Delta \Omega_m = \Delta \text{FP}_m \cdot S_{\text{pkt}}$$
- **UAV Battery Reserves Conserved:**  
  Each alarm triggers a $T_{\text{loiter}} \in [15, 30]\text{ s}$ confirmation maneuver at hover power $P_{\text{hover}} \approx 280\text{ W}$ ($4.2 - 8.4\text{ kJ}$ per event):
  $$\Delta E_m = \Delta \text{FP}_m \cdot P_{\text{hover}} \cdot T_{\text{loiter}}$$
- **Added Search Flight Time:**
  $$\Delta t_{\text{flight}} = \frac{\Delta E_m}{P_{\text{hover}}} = \Delta \text{FP}_m \cdot T_{\text{loiter}}$$

---

## Dataset Corpus & Harvesting

### Controlled 2×2 Experimental Design

The study is frozen around two wilderness environments and two visual modalities:
1. **Arid Desert:** High ambient temperature, rocky terrain, sparse shrubs, and midday thermal crossover where ground rock temperatures equilibrate with human skin radiation.
2. **Temperate Forest:** Dense tree canopies, dappled ground shadows, intermittent foliage occlusions, and wildlife distractors.

Each biome contains three scenario types:
- **Positive Scenarios:** Human targets (walking, crawling, partially occluded).
- **Hard-Negative Scenarios:** Challenging non-target distractors (sun-heated boulders, moving canopy shadows, thermal clutter).
- **Clear-Negative Scenarios:** Empty wilderness terrain without human or distractor targets.

<div align="center">

| Modality / Architecture | Arid Desert | Temperate Forest | Total Harvested Clips | Total Decimated Frames |
| :--- | :---: | :---: | :---: | :---: |
| **Visible RGB (YOLO11n-RGB)** | 30 clips (10/10/10) | 30 clips (10/10/10) | 60 clips | 4,800 frames |
| **Thermal TIR (YOLO11n-Thermal)** | 30 clips (10/10/10) | 30 clips (10/10/10) | 60 clips | 4,800 frames |
| **Total (Aligned Pairs)** | **60 clips** | **60 clips** | **120 streams** | **9,600 frames** |

</div>

*(10/10/10 = 10 Positive / 10 Hard-Negative / 10 Clear-Negative clips per biome).*

---

### Dual-Crop Harvesting Procedure

Raw video sequences are captured in $1280 \times 720$ resolution at 24 FPS (10.0 s duration, 240 frames). To preserve optical ground sampling distance (GSD) without downscaling distortion:
- Two disjoint square snippets of $640 \times 640$ pixels are harvested from each video:
  - **Left Tile:** $x \in [0, 640]$, centered vertically at $y = 40$ px.
  - **Right Tile:** $x \in [640, 1280]$, centered vertically at $y = 40$ px.
- Automated via FFmpeg with `-movflags +faststart` to relocate the `moov` atom to offset 36 for rapid web-based scrubbing in Label Studio.
- Ground-truth keyframe bounding boxes were placed at target inflection points with linear track interpolation. Hard-negative and clear-negative frames are explicitly recorded with empty 0-byte label files.

---

### Decimation & Sequence-Level 60/20/20 Split

Videos are decimated $3:1$ to an operating rate of $f_s = 8\text{ Hz}$ ($T_s = 125\text{ ms}$, 80 frames per snippet). To prevent temporal data leakage, partitioning is performed strictly at the sequence level:

<div align="center">

| Partition | Proportion | Clips per Scenario Bin | Clips per Biome | Total Video Clips | Frames per Modality (8 Hz) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Split** | 60% | 6 clips | 18 clips | 36 clips | 2,880 frames |
| **Validation Split** | 20% | 2 clips | 6 clips | 12 clips | 960 frames |
| **Test Split (Held-Out)** | 20% | 2 clips | 6 clips | 12 clips | 960 frames |
| **Total Corpus** | **100%** | **10 clips** | **30 clips** | **60 clips** | **4,800 frames** |

</div>

---

### Dataset Composition Matrix

<div align="center">

| Biome Condition | Scenario Type | RGB Video Clips | Thermal Video Clips | Frames per Modality (8 Hz) | Sequence Split (Train / Val / Test) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Arid Desert** | Positive Target | 10 | 10 | 800 | 6 / 2 / 2 clips |
| **Arid Desert** | Hard-Negative Distractor | 10 | 10 | 800 | 6 / 2 / 2 clips |
| **Arid Desert** | Clear-Negative Background | 10 | 10 | 800 | 6 / 2 / 2 clips |
| **Temperate Forest** | Positive Target | 10 | 10 | 800 | 6 / 2 / 2 clips |
| **Temperate Forest** | Hard-Negative Distractor | 10 | 10 | 800 | 6 / 2 / 2 clips |
| **Temperate Forest** | Clear-Negative Background | 10 | 10 | 800 | 6 / 2 / 2 clips |
| **Corpus Total** | **All 6 Scenario Bins** | **60** | **60** | **4,800** | **36 / 12 / 12 clips** |

</div>

---

## Results & Benchmarks

> [!NOTE]
> In accordance with research reproducibility standards, empirical detector benchmark values pending experimental hardware runs are preserved as `[TBD]`. No values are fabricated or filled with synthetic guesses.

### Table 1: Upstream Frame-Level Detection Performance
*Evaluated frame-by-frame on held-out test sequences ($N = 960$ frames per stream) at $\tau = 0.50$ prior to temporal post-processing:*

<div align="center">

| Configuration / Condition | Modality | Test Environment | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n-RGB** | Visible RGB | Arid Desert | [TBD] | [TBD] | [TBD] |
| **YOLO11n-RGB** | Visible RGB | Temperate Forest | [TBD] | [TBD] | [TBD] |
| **YOLO11n-Thermal** | Thermal TIR | Arid Desert | [TBD] | [TBD] | [TBD] |
| **YOLO11n-Thermal** | Thermal TIR | Temperate Forest | [TBD] | [TBD] | [TBD] |
| **Late Fusion Gate ($s[n] = \max$)** | Multimodal | Arid Desert | [TBD] | [TBD] | [TBD] |
| **Late Fusion Gate ($s[n] = \max$)** | Multimodal | Temperate Forest | [TBD] | [TBD] | [TBD] |

</div>

---

### Table 2: Comparative Benchmark of Causal Post-Processing
*Comparative evaluation across causal post-processing methods ($W = 5$ frames / 625 ms latency budget) evaluated under validation-optimized thresholds $\tau_m^*$ on held-out test sequences ($N = 960$ frames, 12 clips):*

<div align="center">

| Method | Val $\tau^*$ | Precision | Recall | F1-Score | F1 (95% CI)$^*$ | Pos. Recall (%) | Hard-Neg FP | Clear-Neg FP | FASR (%) | Latency / Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Baseline Raw Thresholding** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M2: Five-Frame Moving Average** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M3: Five-Frame Median Filter** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M4: Five-Frame History Consensus**| [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M5: Learned Mamba-SSSM** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

</div>

$^*$*Two-sided 95% percentile confidence intervals computed via sequence-level bootstrapping ($B = 1000$ iterations).*

---

### Table 3: Environment-Stratified Operational Resource Impact
*Projected mission-level resource savings across held-out test sequences relative to baseline raw thresholding (M1):*

<div align="center">

| Method | Environment | FASR (%) | FP Avoided vs. M1 | Bandwidth Saved (kB)$^\dagger$ | Battery Saved (kJ)$^\ddagger$ | Est. Added Flight Time (min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M2: Moving Average** | Arid Desert | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M2: Moving Average** | Temperate Forest | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M3: Median Filter** | Arid Desert | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M3: Median Filter** | Temperate Forest | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M4: History Consensus** | Arid Desert | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M4: History Consensus** | Temperate Forest | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M5: Mamba-SSSM** | Arid Desert | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **M5: Mamba-SSSM** | Temperate Forest | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

</div>

$^\dagger$*Evaluated at $S_{\text{pkt}} = 1.2\text{ kB}$ standard telemetry ($45\text{ kB}$ for visual thumbnail).*  
$^\ddagger$*Evaluated at $P_{\text{hover}} \approx 280\text{ W}$, $T_{\text{loiter}} = 20\text{ s}$ average.*

---

## Quick Reproduction

### 1. Environment Setup
Clone the repository and install dependencies in a virtual environment:
```bash
git clone https://github.com/IamOumarIbrahim/multi-modal-detection.git
cd multi-modal-detection
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dual-Crop Snippet Harvesting
Execute the automated dual-crop extraction pipeline to harvest $640 \times 640$ snippets from raw $1280 \times 720$ video clips:
```bash
python scripts/harvest_dual_crop_snippets.py
```

### 3. Upstream Detector Training
Train the modality-specific YOLO11n baselines in full FP32 precision with batch size 16:
```bash
# Train YOLO11n-RGB
yolo detect train data=configs/data_rgb.yaml model=yolo11n.pt imgsz=640 epochs=100 batch=16 amp=False optimizer=SGD lr0=0.01 device=0

# Train YOLO11n-Thermal
yolo detect train data=configs/data_thermal.yaml model=yolo11n.pt imgsz=640 epochs=100 batch=16 amp=False optimizer=SGD lr0=0.01 device=0
```

### 4. Validation Threshold Sweep & Post-Processing Benchmark
Run the validation threshold optimization sweep ($\tau \in [0.05, 0.95], \Delta \tau = 0.02$) and evaluate post-processing filters on the held-out test split:
```bash
# Farm optimal thresholds tau_m* on validation set
python scripts/optimize_thresholds.py --val-manifest data/val_manifest.json

# Run comparative benchmark with B=1000 sequence bootstrapping
python scripts/evaluate_benchmark.py --test-manifest data/test_manifest.json --bootstrap 1000
```

---

## Repository Organization

```
multi-modal-detection/
├── annotations/                          # Label Studio configuration and annotation schemas
│   ├── label_studio_config.xml           # Bounding box configuration for Label Studio
│   └── label_studio_video_config.xml     # Video scrub annotation schema
├── configs/                              # Training and dataset configuration templates
│   ├── data.template.yaml                # Ultralytics dataset specification template
│   └── hyperparams.yaml                  # Model training hyperparameters
├── data/                                 # MMSAR multimodal benchmark corpus
│   ├── raw/                              # Harvested dual-crop video snippets (640x640)
│   │   ├── desert/                       # Arid Desert biome (pos, hard_neg, clear_neg)
│   │   └── forest/                       # Temperate Forest biome (pos, hard_neg, clear_neg)
│   └── manifest.json                     # Sequence-level dataset split manifest
├── docs/manuscript/                      # IEEE conference manuscript source and figures
│   ├── figures/                          # System architecture, video snippets, and benchmark plots
│   ├── figure_system_architecture.tex    # Full TikZ system pipeline architecture diagram
│   ├── figure_temporal_postprocessing.tex# TikZ temporal signal post-processing diagram
│   ├── figure_mamba_sssm.tex             # TikZ Mamba selective SSM state transition diagram
│   ├── references.bib                    # BibTeX references
│   └── main.tex                          # Primary IEEE conference LaTeX manuscript
├── scripts/                              # Dataset harvesting and evaluation scripts
│   ├── harvest_dual_crop_snippets.py     # Dual-crop FFmpeg extraction script (640x640)
│   ├── export_and_process_desert_positive.py # Video decimation and label export pipeline
│   └── process_negatives.py              # Zero-byte empty annotation generator for negatives
├── tools/                                # Operational and automation tooling
│   └── generate_architecture_figure.py   # Vector-to-raster compilation tooling
├── CHANGELOG.md                          # Itemized audit and manuscript enhancement changelog
├── DATASET_PROMPTS.md                    # Physics-grounded synthetic video generation prompts
├── PROMISE_AUDIT.md                      # Audit of manuscript claims, forward promises, and figures
├── REMAINING_TBD_LEDGER.md               # Ledger of all remaining TBD cells and hardware resolutions
└── README.md                             # Project overview, architecture, and benchmark documentation
```

---

## Authors & Citation

- **Oumar Mamoun Ibrahim** — Department of Computer Engineering, University of Sharjah, UAE  
  [U22200741@sharjah.ac.ae](mailto:U22200741@sharjah.ac.ae) · [ORCID: 0009-0008-0312-1605](https://orcid.org/0009-0008-0312-1605)
- **Dr. Mohamad Khairi bin Ishak** — Department of Computer Engineering, University of Sharjah, UAE  
  [mishak@sharjah.ac.ae](mailto:mishak@sharjah.ac.ae) · [ORCID: 0000-0002-3554-0061](https://orcid.org/0000-0002-3554-0061)

If you use this work, codebase, or dataset in your research, please cite:

```bibtex
@inproceedings{ibrahim2026lightweight,
  title     = {Lightweight Multimodal Person Detection and Temporal Post-Processing for Aerial Search and Rescue},
  author    = {Ibrahim, Oumar Mamoun and bin Ishak, Mohamad Khairi},
  booktitle = {Proceedings of the 10th International Conference on Signal Processing and Integrated Networks (ICSPIS)},
  year      = {2026},
  address   = {Sharjah, United Arab Emirates},
  month     = {September}
}
```

---

## Acknowledgments & License

This research builds upon [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [Label Studio](https://github.com/HumanSignal/label-studio).  
Code and dataset scripts are licensed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their respective licenses.