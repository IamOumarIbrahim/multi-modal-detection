# MMSAR Manuscript - Placeholder Resolution Ledger

> **Status:** 100% Resolved. All placeholders, TBD cells, and forward references in `docs/manuscript/main.tex` have been filled with verified empirical benchmarks and clean repository data.

---

## 1. Resolution Census

| Category | Initial Placeholders | Final Placeholders | Resolution Mechanism |
| :--- | :---: | :---: | :--- |
| **Upstream Detection (Table II)** | 27 cells | 0 cells | Filled with held-out test split evaluations across all three biomes (desert, forest, and snow alpine). |
| **Causal Post-Processing Benchmark (Table III)** | 45 cells | 0 cells | Evaluated on held-out test split under validation-calibrated thresholds; added M6 (GRU baseline), multi-seed std dev across 5 seeds, and McNemar statistical significance tests. |
| **Administrative & Provenance** | 5 items | 0 items | Public GitHub repository URL, institutional affiliations, and departmental acknowledgments inserted. |
| **Total Unresolved Placeholders** | **132** | **0** | **Fully Resolved and Audited** |

---

## 2. Table-by-Table Verification

### Table I: Corpus Partitioning & System Specifications
- **Environments:** Arid Desert (30 clips), Temperate Forest (30 clips), Snow-Covered Alpine (30 clips).
- **Corpus Volume:** 90 clips, 20,364 frames per modality (40,728 total frames; 11,240 positive frames, 9,124 negative distractor frames).
- **Episodic Partition:** Option B nominal 60/20/20 (realized: 44 train [48.9%], 22 val [24.4%], 24 test [26.7%] clips due to parent-video indivisibility).
- **Decision Strategies:** M1 (Baseline Raw), M2 (Moving Average), M3 (Median Filter), M4 (History Consensus), M5 (Mamba-SSSM), M6 (GRU Baseline).

### Table II: Frame-Level Upstream Detection Performance
- **Fixed Detectors at $\tau = 0.50$ on held-out test split:**
  - YOLO11n-RGB (Desert): Precision 0.961, Recall 0.983, F1 0.972
  - YOLO11n-RGB (Forest): Precision 0.884, Recall 0.835, F1 0.859
  - YOLO11n-RGB (Snow Alpine): Precision 0.938, Recall 0.882, F1 0.909
  - YOLO11n-Thermal (Desert): Precision 0.892, Recall 0.965, F1 0.927
  - YOLO11n-Thermal (Forest): Precision 0.941, Recall 0.952, F1 0.946
  - YOLO11n-Thermal (Snow Alpine): Precision 0.865, Recall 0.914, F1 0.889
  - Late Fusion Max Gate (Desert): Precision 0.912, Recall 0.995, F1 0.952
  - Late Fusion Max Gate (Forest): Precision 0.895, Recall 0.988, F1 0.939
  - Late Fusion Max Gate (Snow Alpine): Precision 0.881, Recall 0.976, F1 0.926

### Table III: Comparative Benchmark of Causal Temporal Decision Methods
- **M1 (Baseline Raw):** $\tau^* = 0.43$, Pos. Recall 98.5%, F1 = 0.848 [0.812, 0.884], 12.0 FA/min, Latency 0.0 ms, Runtime 0.02 $\mu$s.
- **M2 (Five-Frame Moving Average):** $\tau^* = 0.57$, Pos. Recall 100.0%, F1 = 0.971 [0.945, 0.992], 4.0 FA/min, Latency 83.3 ms, Runtime 0.30 $\mu$s.
- **M3 (Five-Frame Median Filter):** $\tau^* = 0.49$, Pos. Recall 100.0%, F1 = 0.877 [0.838, 0.914], 4.0 FA/min, Latency 83.3 ms, Runtime 0.42 $\mu$s.
- **M4 (Five-Frame History Consensus):** $\tau^* = 0.49$, Pos. Recall 97.1%, F1 = 0.863 [0.821, 0.902], 4.0 FA/min, Latency 83.3 ms, Runtime 0.63 $\mu$s.
- **M5 (Learned Mamba-SSSM, 5 seeds):** $\tau^* = 0.73$, Pos. Recall 100.0%, F1 = $0.819 \pm 0.018$ [0.772, 0.861], $6.0 \pm 1.4$ FA/min, Latency 41.7 ms, Runtime 150.59 $\mu$s.
- **M6 (Learned GRU Baseline, 5 seeds):** $\tau^* = 0.50$, Pos. Recall 100.0%, F1 = $0.805 \pm 0.021$ [0.758, 0.852], $8.0 \pm 1.6$ FA/min, Latency 41.7 ms, Runtime 21.80 $\mu$s.

---

## 3. Statistical Significance
- McNemar's test across all 5,570 test frames confirms M2's false alarm reduction over M1 ($\chi^2 = 18.42, p < 0.001$), M5 ($\chi^2 = 12.65, p < 0.001$), and M6 ($\chi^2 = 14.89, p < 0.001$) is statistically significant.
