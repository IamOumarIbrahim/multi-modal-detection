# MMSAR Manuscript - Remaining TBD & Placeholder Ledger

This ledger inventories every `[TBD]`, `[TODO]`, and `[PLACEHOLDER: ...]` remaining in `docs/manuscript/main.tex` following the zero-new-experiment enhancement pass. For each item, the table identifies its manuscript location, the variable or metric represented, and the exact real-world engineering or experimental action required to resolve it.

---

## Summary Census

- **Causal Post-Processing Benchmark Table (Table I / `tab:postprocessing_benchmark`)**: 45 `[TBD]` cells (5 methods $\times$ 9 metrics: $\tau_m^*$, Precision, Recall, F1, F1 95% CI, Positive Recall %, Negative FP, FASR %, Latency/Frame).
- **Upstream Frame-Level Detection Table (Table III / `tab:fusion_comparison`)**: 27 `[TBD]` cells (9 conditions (added Snow/Alpine for RGB, Thermal, and Fusion) $\times$ 3 metrics: Precision, Recall, F1-Score at $\tau = 0.50$).
- **Environment-Stratified Operational Impact Table (Table IV / `tab:operational_impact`)**: 60 `[TBD]` cells (4 methods $\times$ 3 environments $\times$ 5 metrics: FASR %, FP Avoided, Bandwidth Saved in kB, Battery Saved in kJ, Added Flight Time in min).
- **Administrative placeholders**: resolved (previously 5 items — Data/Code Availability and Acknowledgments sections now contain final text)

**Total Unresolved Cells / Placeholders: 45 + 27 + 60 = 132 empirical table cells** - every item strictly belongs to experimental evaluation, physical hardware measurement, or administrative provenance.

---

## Itemized Ledger

### 1. Causal Post-Processing Comparative Benchmark (Table I: `tab:postprocessing_benchmark`)

| Method Row | Target Metric Columns | Exact Real-World Resolution Action |
|---|---|---|
| **M1: Baseline Raw Thresholding** | Val $\tau^*$, Precision, Recall, F1, F1 (95% CI), Pos. Recall %, Negative FP, FASR %, Latency/Frame | Sweep $\tau \in [0.05, 0.95]$ (step 0.02) on the validation clips to identify $\tau_{\text{raw}}^*$ maximizing F1; apply $\tau_{\text{raw}}^*$ to the test clips; run $B=1000$ clip bootstraps for 95% CI; log per-condition FP and frame execution time. |
| **M2: Five-Frame Moving Average** | Val $\tau^*$, Precision, Recall, F1, F1 (95% CI), Pos. Recall %, Negative FP, FASR %, Latency/Frame | Sweep $\tau$ on validation clips over $\tilde{s}_{\text{MA}}[n]$ to find optimal $\tau_{\text{MA}}^*$; execute causal 5-frame moving average on test clips; compute bootstrapped CI, negative FP counts, FASR vs. M1 baseline, and filter execution latency. |
| **M3: Five-Frame Median Filter** | Val $\tau^*$, Precision, Recall, F1, F1 (95% CI), Pos. Recall %, Negative FP, FASR %, Latency/Frame | Sweep $\tau$ on validation clips over order-statistic $\tilde{s}_{\text{med}}[n]$ to find $\tau_{\text{med}}^*$; filter test streams with sliding median window; compute bootstrapped CI, FP counts, FASR, and runtime latency. |
| **M4: Five-Frame History Consensus** | Val $\tau^*$, Precision, Recall, F1, F1 (95% CI), Pos. Recall %, Negative FP, FASR %, Latency/Frame | Sweep $\tau$ on validation clips under 3-of-5 voting to find $\tau_{\text{hist}}^*$; execute discrete boolean voting window on test clips; compute bootstrapped CI, FP counts, FASR, and latency. |
| **M5: Learned Mamba-SSSM** | Val $\tau^*$, Precision, Recall, F1, F1 (95% CI), Pos. Recall %, Negative FP, FASR %, Latency/Frame | Train `MambaSSSMModel` on the training sequences using BCE loss for 50 epochs; sweep $\tau$ on the validation sequences to select $\tau_{\text{Mamba}}^*$; evaluate frozen weights on the test sequences; compute bootstrapped CI, FP counts, FASR, and PyTorch recurrent forward time. |

### 2. Upstream Frame-Level Detection (Table III: `tab:fusion_comparison`)

| Row / Condition | Target Metric Columns | Exact Real-World Resolution Action |
|---|---|---|
| YOLO11n-RGB (Arid Desert) | Precision, Recall, F1-Score | Run YOLO11n-RGB at $\tau = 0.50$ on the held-out Arid Desert test clips (positive and negative episodes); compute TP, FP, FN against annotated person bounding boxes. |
| YOLO11n-RGB (Temperate Forest) | Precision, Recall, F1-Score | Run YOLO11n-RGB at $\tau = 0.50$ on the held-out Temperate Forest test clips; compute TP, FP, FN against annotated person bounding boxes. |
| YOLO11n-Thermal (Arid Desert) | Precision, Recall, F1-Score | Run YOLO11n-Thermal at $\tau = 0.50$ on the held-out Arid Desert test clips; compute TP, FP, FN against annotated person bounding boxes. |
| YOLO11n-Thermal (Temperate Forest) | Precision, Recall, F1-Score | Run YOLO11n-Thermal at $\tau = 0.50$ on the held-out Temperate Forest test clips; compute TP, FP, FN against annotated person bounding boxes. |
| Late Fusion Gate (Desert) | Precision, Recall, F1-Score | Evaluate decision fusion $s[n] = \max(c_{\text{RGB}}[n], c_{\text{Thermal}}[n])$ at $\tau = 0.50$ across Arid Desert test clips; log combined TP, FP, FN. |
| Late Fusion Gate (Forest) | Precision, Recall, F1-Score | Evaluate decision fusion $s[n] = \max(c_{\text{RGB}}[n], c_{\text{Thermal}}[n])$ at $\tau = 0.50$ across Temperate Forest test clips; log combined TP, FP, FN. |

### 3. Operational Resource Conservation (Table IV: `tab:operational_impact`)

| Method & Environment Row | Columns: FASR, FP Avoided, Bandwidth Saved, Battery Saved, Added Flight Time | Exact Real-World Resolution Action |
|---|---|---|
| M2 (Moving Average) - Arid Desert | FASR, $\Delta\text{FP}$, Bandwidth (kB), Battery (kJ), Flight Time (min) | Count false positives on Arid Desert test clips for M2; subtract from M1 Arid Desert baseline; multiply $\Delta\text{FP}$ by $S_{\text{pkt}} = 1.2\text{ kB}$ and $P_{\text{hover}} T_{\text{loiter}} = 280\text{ W} \times 20\text{ s} = 5.6\text{ kJ}$; divide energy saved by 280 W. |
| M2 (Moving Average) - Temperate Forest | FASR, $\Delta\text{FP}$, Bandwidth (kB), Battery (kJ), Flight Time (min) | Count false positives on Temperate Forest test clips for M2; compute operational savings relative to M1 Forest baseline using identical multipliers. |
| M3 (Median Filter) - Arid Desert | FASR, $\Delta\text{FP}$, Bandwidth (kB), Battery (kJ), Flight Time (min) | Count false positives on Arid Desert test clips for M3; apply resource conversion formulas. |
| M3 (Median Filter) - Temperate Forest | FASR, $\Delta\text{FP}$, Bandwidth (kB), Battery (kJ), Flight Time (min) | Count false positives on Temperate Forest test clips for M3; apply resource conversion formulas. |
| M4 (History Consensus) - Arid Desert | FASR, $\Delta\text{FP}$, Bandwidth (kB), Battery (kJ), Flight Time (min) | Count false positives on Arid Desert test clips for M4; apply resource conversion formulas. |
| M4 (History Consensus) - Temperate Forest | FASR, $\Delta\text{FP}$, Bandwidth (kJ), Battery (kJ), Flight Time (min) | Count false positives on Temperate Forest test clips for M4; apply resource conversion formulas. |
| M5 (Mamba-SSSM) - Arid Desert | FASR, $\Delta\text{FP}$, Bandwidth (kB), Battery (kJ), Flight Time (min) | Count false positives on Arid Desert test clips for M5; apply resource conversion formulas. |
| M5 (Mamba-SSSM) - Temperate Forest | FASR, $\Delta\text{FP}$, Bandwidth (kB), Battery (kJ), Flight Time (min) | Count false positives on Temperate Forest test clips for M5; apply resource conversion formulas. |

### 4. Administrative & Provenance Placeholders

| Location | Item / Placeholder | Exact Real-World Resolution Action |
|---|---|---|
| Data & Code Availability | `[PLACEHOLDER: code repository URL, ...]` | Insert public GitHub repository link (e.g., `https://github.com/IamOumarIbrahim/multi-modal-detection`) upon camera-ready clearance. |
| Data & Code Availability | `[PLACEHOLDER: dataset repository URL or DOI]` | Upload harvested synthetic dual-crop video clips and Label Studio JSON manifests to Zenodo or Kaggle Datasets; insert permanent DOI. |
| Data & Code Availability | `[PLACEHOLDER: model weight repository URL]` | Upload pre-trained YOLO11n-RGB, YOLO11n-Thermal, and Mamba-SSSM `.pt` weight checkpoints to Hugging Face Hub or GitHub Releases; insert direct URL. |
| Acknowledgment | `[PLACEHOLDER: funding agency / grant number]` | Insert University of Sharjah competitive research grant ID and sponsor agency details following institutional sign-off. |
| Acknowledgment | `[PLACEHOLDER: institutional or lab contributors]` | Insert departmental laboratory names and computing facility acknowledgments. |
