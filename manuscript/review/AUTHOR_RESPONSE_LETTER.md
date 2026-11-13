# Author Response & Detailed Revision Notes
**IEEE ICSPIS 2026 Paper #1571354510**  
**Title:** *Causal Temporal Filtering for Reliable Alerts in Dual-Stream UAV Search and Rescue Detection*  
**Authors:** Oumar Mamoun Ibrahim and Mohamad Khairi Bin Ishak  
**Affiliation:** Department of Computer Engineering, University of Sharjah, UAE  

---

Dear Editors and Reviewers,

We sincerely thank the Technical Program Committee and the reviewers for their constructive, insightful, and rigorous feedback on our manuscript. We have carefully addressed every critique, resulting in substantial improvements to the clarity, technical precision, statistical rigor, and transparency of our paper.

Below is our point-by-point response detailing the specific revisions incorporated into the camera-ready manuscript.

---

## Response to Reviewer 1

### 1. Pseudo-TIR Characterization & Genuine Spectral Complementarity
> **Reviewer Critique:**  
> *"The pseudo-TIR channel is a 256-level LUT remap of the same synthetic RGB frames, so both streams are deterministic functions of identical pixels. A LUT cannot recover information that shadow, saturation, or occlusion destroyed in the source frame, yet Section V.A argues that fusion 'compensates for complementary dropouts' when one sensor is 'blinded by environmental interference.' The per-stream differences in Table II reflect two detectors trained on differently colorized copies of one image set, not genuine spectral complementarity."*

**Author Response & Revisions:**  
We completely agree with the reviewer. We have thoroughly reframed the paper to eliminate all claims of physical thermal emissivity, radiometric thermal crossover, or genuine multispectral complementarity. The experiment is now strictly characterized as the fusion of **two correlated edge representations derived from identical pixels**:
1. **Introduction (Section I, Paragraph 2 & Section V.F):** We state explicitly:
   > *"Because this pseudo-TIR channel is deterministically derived from RGB luminance, it contains no independent physical thermal or spectral information, and a LUT cannot restore details lost to shadow or saturation. Rather, it creates a non-linear luminance remapping that induces distinct feature representations in convolutional detectors... We treat this dual-stream setup as a controlled testbed for fusing two correlated edge representations under causal temporal filtering."*
2. **System Architecture Figure (Fig. 1):** The block diagram labels have been corrected from "Thermal Stream" to *"Grayscale Luminance Remap (256-level LUT)"* and *"Pseudo-TIR Color Transfer"*.
3. **Upstream Detection Analysis (Section V.A):** We explicitly clarify that:
   > *"The per-stream differences in Table II reflect two detectors trained on differently colorized representations of the same source imagery rather than physical multispectral complementarity. The grayscale LUT alters intensity contrast and removes chrominance, yielding distinct detection errors... Late fusion max-pooling pools confident detections from either detector... However, because max-pooling forwards the higher score, it also propagates isolated noise spikes into downstream buffers, motivating causal temporal post-processing."*
4. **Limitations (Section V.F):** Point 1 explicitly bounds our findings:
   > *"First, the pseudo-TIR channel is produced deterministically from RGB frames via a 256-level grayscale lookup table rather than radiometric sensors; discussions of thermal contrast characterize non-linear luminance remapping rather than physical emissivity or real thermal crossover."*

---

### 2. Sample Size, Bootstrap CIs, Flight Time Denominators & Multiple-Comparison Corrections
> **Reviewer Critique:**  
> *"With 13 positive tiles in the held-out split, the 92.3% versus 84.6% recall difference is one tile, and the biome-stratified rates in Table IV rest on 4–5 tiles each. A single p-value (0.4922, M2 vs M5) is reported to support a claim of no significant gain that is made across all learned-versus-classical comparisons. Please report the full set of tests with a multiple-comparison correction, state the flight time underlying the FA/min denominators, and consider bootstrap confidence intervals on alert recall so readers can see how wide they are."*

**Author Response & Revisions:**  
We thank the reviewer for requesting rigorous statistical treatment. We have implemented all requested items:
1. **Single-Tile Difference & Bootstrap CIs:** In Table III and Section V.B, we report 95% non-parametric bootstrap confidence intervals (10,000 resamples):
   - M1 / M3 / M4 (12/13 positive tiles): **92.3% [76.9%, 100.0%]**
   - M2 (11/13 positive tiles): **84.6% [61.5%, 100.0%]**
   - The text explicitly notes that the 92.3% vs. 84.6% difference represents exactly **one tile** (12/13 vs. 11/13) and that the confidence intervals are wide and heavily overlapping.
2. **Independent Parent-Video Recall:** To eliminate sibling-crop dependency, we evaluate recall at the independent parent-video level ($N=9$ positive parent videos in the test split). Across all four classical methods (M1, M2, M3, M4), parent-video recall is **identical at 88.9% (8/9 parent videos; 95% CI [66.7%, 100.0%])**, as the single tile missed by M2 has an alerted sibling crop from the same sweep.
3. **Full Pairwise Wilcoxon Tests with Holm-Bonferroni Correction:** Section V.C now reports the complete set of pairwise tests:
   - *Parent-video level ($N=12$):*
     - M2 vs. M5: $W=7.0$, uncorrected $p=0.1484$, Holm-Bonferroni adjusted $p=0.3906$
     - M2 vs. M6: $W=3.0$, uncorrected $p=0.0781$, Holm-Bonferroni adjusted $p=0.3906$
     - M3 vs. M5: $W=5.0$, uncorrected $p=0.0781$, Holm-Bonferroni adjusted $p=0.3906$
     - M3 vs. M6: $W=9.0$, uncorrected $p=0.4688$, Holm-Bonferroni adjusted $p=0.4688$
   - *Episode level ($N=24$):* Uncorrected differences in $F_1$ (e.g., M2 vs. M5 $p=0.0522$, M3 vs. M5 $p=0.0342$, M2 vs. M6 $p=0.0488$) all become statistically non-significant after Holm-Bonferroni adjustment (all adjusted $p \ge 0.1709$).
4. **Flight Time Denominators:** We explicitly report $T_{\text{flight}} = 3.868\text{ min}$ (232.1 s across 5,570 frames at 24 Hz), broken down by biome in Table IV and Section III.D: Desert 1.333 min, Forest 1.201 min, Snow Alpine 1.333 min.
5. **Per-Biome Positive Tile Caveats:** In Section V.D, we state: *"Because each biome contributes only 4–5 positive tiles (desert: 4 positive, 4 negative; forest: 4 positive, 4 negative; snow: 5 positive, 3 negative), per-biome recall differences must be interpreted with caution: 75.0% recall in forest reflects 3 of 4 tiles detected, and 80.0% in snow reflects 4 of 5 tiles."*

---

### 3. Upstream Latency Profiling & Edge Budget Framing
> **Reviewer Critique:**  
> *"The 73×–500× runtime gap is measured between filters costing 0.3 µs and 151 µs, against a frame budget of 41.7 ms (Eq. 1) dominated by two YOLO11n forward passes that are never timed. Even the slowest filter consumes roughly 0.4% of that budget, so the practical case against learned models is memory footprint and the need for training data rather than latency. Please report T_detect and T_fusion so Eq. (1) can actually be checked, and adjust the framing accordingly."*

**Author Response & Revisions:**  
We appreciate this practical insight and have thoroughly reframed our edge arguments:
1. **Measured Component Runtimes:** In Section I, Section III.A (Eq. 1), and Section V.C, we report the full breakdown:
   - Dual parallel YOLO11n passes: $T_{\text{detect}} \approx 10.2\text{ ms}$ on GPU ($\sim 40.0\text{ ms}$ on CPU).
   - Late fusion max gate: $T_{\text{fusion}} < 0.001\text{ ms}$.
   - Filter execution: $T_{\text{filter}} \in [0.02\,\mu\text{s}, 150.59\,\mu\text{s}]$.
   - Total pipeline latency: $10.35\text{ ms} \ll 41.7\text{ ms}$.
2. **Reframed Case Against Learned Models:** We explicitly acknowledge that all filters consume $<0.4\%$ of the 41.7 ms frame budget. As stated in Section I and Section V.C:
   > *"The definitive argument against learned sequence models on this edge task is therefore not frame deadline violations, but rather their memory overhead (512 B state vs. 20 B), training pipeline, hyperparameter sensitivity, and lack of demonstrable alert gain over lightweight, training-free causal filtering."*

---

## Response to Reviewer 2

### 1. Embedded UAV Testing & Real RGB-Thermal Data
> **Reviewer Critique:**  
> *"The paper would be stronger with testing on real RGB–thermal data and embedded UAV hardware, since the current experiments rely mainly on synthetic videos and pseudo-thermal images. The relatively small test set should also be considered when interpreting the results."*

**Author Response & Revisions:**  
We acknowledge these constraints and have made them central to the paper's scope:
1. **Scope and Limitations (Section V.F):** We added dedicated discussion items detailing that synthetic video lacks live aerodynamic turbulence, gust disturbances, and live human thermal emissivity.
2. **Embedded Hardware Deployment Roadmap:** In the Abstract, Section V.F, and Conclusion (Section VI), we state that desktop GPU execution provides a baseline and that embedded flight trials on the NVIDIA Jetson Orin Nano under aerodynamic turbulence are planned for future field studies.
3. **Small Test Set Interpretation:** We contextualize the 13 positive tiles with bootstrap confidence intervals and parent-video aggregation (Section V.B and Table III).

---

### 2. Dataset Generation and Detector Setup Details
> **Reviewer Critique:**  
> *"Some additional details about the generated data and detector setup would improve reproducibility. Overall, this is a clear and promising contribution with practical value."*

**Author Response & Revisions:**  
We have added full specifications in Table I, Section IV, and the Ethical Statement:
- **Generative AI Models & Prompts:** Google Gemini 3.8 Flash and xAI Grok text-to-video models simulating 30–50 m AGL altitudes and 30–45° oblique depression angles.
- **Corpus Details:** Exactly 45 parent videos (15 desert, 14 forest, 16 snow alpine; 21 positive scenarios with targets occupying 1–4% frame area, 24 negative distractors).
- **Detector Configurations:** Ultralytics YOLO11n (2.6M parameters, 6.5 GFLOPs at $640\times 640$, FP32), batch 16, 60 epochs, SGD cosine scheduler ($\text{lr}_0 = 0.01$, momentum 0.937, weight decay 0.0005), NMS IoU threshold 0.70.
- **Sequence Models:** Mamba-SSSM ($d_{\text{model}}=16, d_{\text{state}}=8$, 753 parameters, 512 B state) and GRU ($d_{\text{hidden}}=16$, 1,681 parameters, 64 B state), trained with Adam ($\text{lr}=0.01$, BCE loss, 50 epochs) across 5 seeds.
- **Code & Manifest Availability:** Full simulation code, split manifests, and PyTorch weights are provided at: [https://github.com/IamOumarIbrahim/multi-modal-detection](https://github.com/IamOumarIbrahim/multi-modal-detection).

---

## Response to Reviewer 3

### 1. Removing Temperature/Emissivity Claims & Reframing
> **Reviewer Critique:**  
> *"The pseudo-TIR stream is produced deterministically from RGB luminance and therefore contains no independent thermal or spectral information. The manuscript should either reframe the experiment as fusion of two correlated RGB-derived representations or validate the proposed approach using genuine paired RGB-thermal data. Claims concerning thermal crossover, temperature contrast, and complementary sensor failures should be removed unless supported by real thermal measurements."*

**Author Response & Revisions:**  
As detailed in our response to Reviewer 1, we have comprehensively excised all unsupported claims:
- All references to "thermal crossover" and "temperature contrast" have been removed.
- The experiment is strictly presented as fusing two correlated RGB-derived representations (one full-color, one non-linear luminance remap via a 256-level LUT: $Y = 0.299R + 0.587G + 0.114B$).
- High snow brightness and sun-heated desert rock detections are explicitly identified as **synthetic luminance artifacts** of the lookup table rather than physical thermal emissions.

---

### 2. Comprehensive Reproducibility Details
> **Reviewer Critique:**  
> *"The authors should also strengthen reproducibility by specifying the exact video-generation models, prompts, seeds, number of parent videos, clip-selection criteria, detector settings, confidence-stream construction, and complete Mamba/GRU training configurations."*

**Author Response & Revisions:**  
We have incorporated every single configuration parameter:
- **Parent Videos & Clips (Table I & Section IV):** 45 parent videos partitioned via Option B parent-grouped episodic partitioning (44 train [17 pos, 27 neg], 22 val [12 pos, 10 neg], 24 test [13 pos, 11 neg] tiles per channel; 5,570 test frames).
- **Confidence-Stream Construction (Eq. 2 & Eq. 3):** Highest bounding box confidence per frame $c[n] = \max_i c_i[n]$ ($0.0$ if empty), merged via late fusion max gate $s[n] = \max(c_{\text{RGB}}[n], c_{\text{Thermal}}[n])$, buffered into causal window $\mathbf{w}[n]$ of length $W=5$.
- **Mamba & GRU Architecture & Training (Section III.C):** Complete equations, state dimensions, parameter counts, loss function, optimizer, learning rate, and seed sweeping protocol (5 seeds).

---

### 3. Timing Definitions Reconciliation
> **Reviewer Critique:**  
> *"Finally, the statistical methodology and timing definitions require reconciliation..."*

**Author Response & Revisions:**  
We have clarified all temporal definitions to remove ambiguity:
1. **Filter Buffer Span vs. Total Integration:** In Section III.C and Fig. 1, we clarify that for $W=5$ at 24 Hz ($T_s \approx 41.7\text{ ms}$), the buffer span between the first and last samples is $T_{\text{span}} = (W-1)T_s = 166.7\text{ ms}$ (with a 5-frame total integration duration $W T_s = 208.3\text{ ms}$).
2. **Alert Latency vs. Filter Buffer Delay:** In Section III.D, we distinguish filter buffer delay (166.7 ms) from operational **Alert Latency (Time-to-Alarm)** (188.9–592.3 ms in Table III), which additionally includes upstream detector confidence ramp-up as targets enter the camera field of view.

---

### 4. Window Length Sensitivity & Operational Regimes
> **Reviewer Critique:**  
> *(Sensitivity analysis and window ablation requirements)*

**Author Response & Revisions:**  
1. **Window-Length Sensitivity (Section V.E):** Swept across $W \in \{1, 3, 5, 7\}$ on validation (22.8, 15.3, 8.4, 5.8 FA/min) and confirmed on test for both moving average (4.65, 3.10, 2.07, 2.33 FA/min) and median filtering (5.69, 4.65, 3.62, 3.10 FA/min).
2. **Recall-Weighted Sensitivity Analysis (Section V.G & Table V):** Separate mathematical formulations for $F_2$ and asymmetric misclassification cost with precision floor $P \ge 0.95$ demonstrate that causal filters protect the system from false-alert saturation when operating under high-recall doctrines.

---

### 5. Ethical Statement & Living Subjects Footnote
In Footnote 1 of the manuscript, we added the formal statement:
> *"Ethical Statement: No living human subjects or animals were imaged; all training and evaluation video streams were synthetically generated using text-to-video generative models. Code, simulation frameworks, and dataset split manifests are publicly available at: \url{https://github.com/IamOumarIbrahim/multi-modal-detection}."*

---

## Manuscript Formatting Compliance Summary
- **Page Length:** Strictly **6 pages** (two-column IEEE conference format, A4 paper size).
- **Column Balance:** Page 6 columns are balanced with references [1]–[14] concluding in Column 2.
- **Fonts & Graphics:** All fonts are Type 1 PostScript scalable fonts, and all figures use vector TikZ / PGFPlots with embedded high-resolution PNG copies, fully compliant with IEEE PDF eXpress guidelines.

We believe these revisions comprehensively address all reviewer comments and look forward to presenting our work at ICSPIS 2026.
