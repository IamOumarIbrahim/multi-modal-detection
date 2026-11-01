# Scripts

Run every script from the repository root after `pip install -e .`.

| Step | Script | Reads | Writes |
| :---: | :--- | :--- | :--- |
| 1 | `harvest_dual_crop_snippets.py` | `videos/{biome}/{scenario}/rgb/*.mp4` | `data/raw/.../rgb/*_left.mp4`, `*_right.mp4` |
| 2 | `convert_raw_rgb_to_thermal.py` | rgb tiles, `melobytes_medium_low_lut.npy` | `data/raw/.../thermal/*.mp4` |
| 3 | `build_frames_and_labels.py` | tiles, `data/annotations/raw_exports/project_2_rgb_annotations_adjusted.json` | `data/processed/{images,labels}/...` |
| 4 | `split_dataset_episodes.py` | tiles, labels | `data/splits/*`, `configs/data_{rgb,thermal}.yaml` |
| 5 | `train_full_detectors.py` | `configs/hyperparams.yaml` | `runs/detect/yolo11n_{rgb,thermal}_60e/` |
| 6 | `extract_confidences.py` | weights, split lists | `runs/detect/unified_confidences_cache.json` |
| 7a | `evaluate_locked_thresholds.py` | cache | `runs/detect/locked_benchmark_results.json` (paper Tables II to IV) |
| 7b | `evaluate_mmsar_unified.py` | cache | `runs/detect/unified_evaluation_results.json` (val-F1 protocol, window ablation) |
| 7c | `fbeta_threshold_sensitivity.py` | cache | `runs/detect/threshold_sensitivity_*.json` (paper Table V) |
| - | `generate_figure_biome_samples.py` | frames, labels | `docs/manuscript/figures/figure_2_biome_samples.png` |

Steps 7a to 7c only need the committed cache and run in about a minute on CPU.
