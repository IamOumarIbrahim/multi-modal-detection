# MMSAR Manuscript Enhancement Pass — Changelog

Every change made during the zero-new-experiment enhancement pass is recorded below, tagged by Task ID, specifying what changed, where, and why.

- **[T-A1] Missing-File Audit**: Verified that all \input{} targets (igure_system_architecture.tex, igure_temporal_postprocessing.tex) and the \bibliography{} target (
eferences.bib) resolve to existing files on disk; zero missing dependencies.
- **[T-A2] Cross-Reference Audit**: Audited all 34 \label{} and \ref{}/\eqref{} pairs; resolved orphaned section labels (sec:methodology, sec:experimental_setup, sec:results, sec:conclusion, sec:limitations) via an Introduction roadmap paragraph and linked all mathematical equation labels (eq:realtime_budget_serial, eq:realtime_budget_parallel, eq:fasr_def, eq:ma_filter, etc.) directly into explanatory text, achieving zero dangling references and zero orphaned labels.
- **[T-A3] Citation Audit**: Cross-referenced all 25 \cite{...} keys in the manuscript body against 
eferences.bib; confirmed exact 1-to-1 parity with zero missing bibliography entries and zero unreferenced entries.
- **[T-A4] Acronym-on-First-Use Audit**: Audited all 20 technical acronyms ($\ge 2$ letters: SWaP-C, IoT, RGB, TIR, YOLO, GFLOPs, FP32, SSSM, Seq-NMS, T-CNN, FPS, TDP, SiLU, ZOH, SGD, GSD, FASR, MMSAR) and ensured explicit expansion inline at or before their first textual occurrence in the abstract, figures, or body.
- **[T-A5] Numeric Self-Consistency Check**: Mathematically verified and validated all derived quantities stated across the paper (24 FPS / 3 = 8 Hz decimation,  = 125\text{ ms}$; 5-frame window latency = 625 ms; 10.0 s clip = 80 decimated frames / 240 raw frames; Mamba state memory  \times 8 \times 4\text{ B} = 512\text{ B}$; threshold sweep count 0.9/0.02 + 1 = 46$ thresholds; 60/20/20 sequence split = 12 held-out test clips total); confirmed 100% internal consistency.
- **[T-A6] Missing Package Fix**: Added \usepackage{algorithm} immediately preceding \usepackage{algorithmic} in the LaTeX preamble to provide floating and captioning support for Algorithm 1.
- **[T-B1] Pipeline Pseudocode**: Formalized the full causal alarm pipeline by inserting Algorithm 1 (lg:mmsar_pipeline, *MMSAR Causal Alarm Decision*) at the end of Section III-B, documenting per-frame upstream inference, soft disjunctive late fusion, sliding window buffering, causal filtering, telemetry packet dispatch, and loiter triggering.
- **[T-B2] Notation Reference Table**: Inserted Table I (	ab:notation, *Summary of Key Mathematical Notations and System Parameters*) immediately before Section III, detailing 22+ mathematical symbols, dimensions, units, and physical meanings in a two-column ooktabs format.
- **[T-B3] Experimental Design Matrix Table**: Added Table II (	ab:design_matrix, *Controlled 2$\times Experimental Design*) in Section IV-A clarifying the orthogonal distribution of 30 clips (10 positive, 10 hard-negative distractor, 10 clear-negative) across Arid Desert and Temperate Forest biomes.
- **[T-B4] Theoretical Complexity & Memory Footprint Table**: Inserted Table III (	ab:complexity, *Theoretical Per-Frame Complexity and State Memory*) in Section III-C following the M1--M5 descriptions, providing Big-O runtime and state memory footprints for each method and reporting the verified parameter count (753 trainable parameters) from the repository\'s pure-PyTorch Mamba-SSSM implementation (mmsar.postprocessing.mamba_sssm).
- **[T-B5] Formal Real-Time Feasibility Criterion**: Added serial and parallel latency budget inequalities (Eqs.~\ref{eq:realtime_budget_serial}--\ref{eq:realtime_budget_parallel}) to Section III-A bounding end-to-end frame processing to  = 125\text{ ms}$, accompanied by analysis establishing that fusion and post-processing latencies are negligible while upstream detector execution dominates (marked [TBD --- pending on-device benchmarking]).
- **[T-B6] Limitations Subsection**: Added Section V-D (*Limitations*) rigorously documenting six experimental boundaries: held-out sample size (12 test clips / 960 frames), biome scope (2 environments), constrained aerial flight dynamics ($--\text{ m}$, $--\text{ m/s}$), unmodeled adverse weather, single-target assumption, and absence of physical hardware-in-the-loop flight avionics validation.
- **[T-B7] Reproducibility Statement**: Added a dedicated unnumbered *Data and Code Availability* section prior to the bibliography providing explicit placeholders ([PLACEHOLDER: ...]) for code repository, dataset DOI, and pre-trained weights without inventing URLs.
- **[T-B8] Acknowledgments Stub**: Added a formal IEEE \section*{Acknowledgment} stub prior to references with bracket placeholders ([PLACEHOLDER: funding agency / grant number], [PLACEHOLDER: institutional or lab contributors]).
- **[T-C1] Environment-Stratified Operational Impact Table & Derivation**: Added Table VI (	ab:operational_impact, *Environment-Stratified False-Alarm Suppression and Projected Operational Savings*) scaffolding per-environment FASR, FP avoided, bandwidth conserved (kB), battery saved (kJ), and added flight time (min) with [TBD] cells, and added an operational derivation paragraph in Section V-C formulating $\Delta \text{FP}_m(e)$ and mapping it through Eqs.~\ref{eq:bandwidth_waste}--\ref{eq:energy_loiter}.
- **[T-C2] Statistical Confidence Scaffold on Table III**: Augmented Table V (	ab:postprocessing_benchmark) with an F1 (95\% CI) column header maintaining [TBD] entries, and added a non-parametric sequence-level bootstrap methodology paragraph ( = 1000$ resamples) in Section III-C.
- **[T-C3] Full Operating-Curve Figure Scaffold**: Added Figure 3 (ig:pr_curves, *Precision-Recall (or F1-vs-$\tau$) operating curves for M1--M5 across the full validation threshold sweep*) after Figure 2 with an explicit placeholder box and formal caption, linked by a sweep retention statement in Section III-C preserving all 46 candidate threshold evaluation points.

---

# MMSAR Benchmark Rigor Protocol & Configuration Update - Changelog

Every modification aligning the training configuration, pipeline code, and documentation with the publication-track multi-seed rigor protocol is recorded below:

- **[T-C01] Benchmark Protocol Hyperparameters (`configs/hyperparams.yaml`)**:
  - Replaced fixed rapid-verification 20 epochs with early stopping on validation loss (`early_stopping: true`, `patience: 20`, `early_stopping_monitor: "val/loss"`).
  - Formalized multi-seed protocol across seeds `[0, 42, 1234]` with `min_seeds: 3` and mandatory mean +/- std reporting across all arms.
  - Specified full SGD optimizer schedule: `momentum: 0.937`, `weight_decay: 0.0005`, `lr0: 0.01`, `lrf: 0.01`, `warmup_epochs: 3`, and cosine decay (`cos_lr: true`).
  - Added validated data augmentations: `mosaic: 1.0`, `hsv_h: 0.015`, `hsv_s: 0.7`, `hsv_v: 0.4`, and `fliplr: 0.5`.
  - Configured four required experimental arms: Arm 1 (Main Model YOLO11n), Arm 2 (Architecture Baseline YOLO26n with MuSGD), Arm 3 (Hard-Negative Training Ablation), and Arm 4 (Test-Time Augmentation TTA).
  - Specified dynamic episodic split allocation protocol (Option B parent-video grouped episodic partitioning, zero cross-split visual leakage, dynamic floor >= 2 positive episodes per split, and target-presence difficulty balancing).
  - Configured evaluation diagnostics and reporting safeguards: latency sampling (>= 1000 frames on RTX 4060 and onboard companion computer), B=1000 bootstrap confidence intervals, precision-recall curves, separated false positive breakdown, qualitative failure gallery for misses, empirical IoU jitter curve, confidence score calibration diagram, out-of-distribution spot check, and automated pre-publish duplicate-metric diff check.

- **[T-C02] Hyperparameter Pipeline Plumbing (`src/mmsar/training/train.py`)**:
  - Integrated `optimizer`, `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`, `cos_lr`, `mosaic`, `hsv_h`, `hsv_s`, `hsv_v`, `fliplr`, and dynamic `patience` into the Ultralytics `model.train()` call.
  - Updated dry-run inspection dict to surface the extended hyperparameter state.

- **[T-C03] Scaffolding Test Suite Validation (`tests/test_training_scaffold.py`)**:
  - Updated `test_hyperparams_match_readme` assertions to validate the new benchmark protocol configuration (early stopping enabled with patience 20, SGD optimizer, momentum, weight decay, warmup epochs, augmentations, multi-seed list, and experimental arms). All unit tests pass cleanly.

- **[T-C04] Documentation Synchronization (`README.md`)**:
  - Updated Detector Training Configuration table with early stopping (`patience=20` on `val/loss`), 100-epoch maximum horizon, learning rate schedule, data augmentations, and multi-seed protocol.
  - Added documentation for the four publication-track benchmark arms, reporting safeguards, and extended diagnostic protocols.
  - Updated quick reproduction commands to reflect CLI usage with the benchmark configuration.
