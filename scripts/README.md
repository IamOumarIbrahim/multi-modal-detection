# Scripts

Run every script from the repository root after `pip install -e ".[dev,figures]"`.

## From the committed cache (CPU, minutes)

| Script | Writes |
| :--- | :--- |
| `reproduce.py` | runs everything below in order; `--verify` fails if any file in `runs/detect/` changes |
| `evaluate_locked_thresholds.py` | `runs/detect/locked_benchmark_results.json` (paper Tables III and IV, fixed τ\*) |
| `evaluate_mmsar_unified.py` | `runs/detect/unified_evaluation_results.json` (τ\* from validation F1, window ablation, τ sweep) |
| `fbeta_threshold_sensitivity.py` | `runs/detect/threshold_sensitivity_*.json` (paper Table V: `--objective f2 --min_precision 0.95`) |
| `make_figures.py` | `figures/fig_*.png`, `figures/demo_*.mp4`, `figures/demo_forest.gif` (frame figures and demos need the local videos and are skipped without them) |

## From the videos (GPU, hours)

| Step | Script | Reads | Writes |
| :---: | :--- | :--- | :--- |
| 1 | `harvest_dual_crop_snippets.py` | `videos/{biome}/{scenario}/rgb/*.mp4` | `data/raw/.../rgb/*_left.mp4`, `*_right.mp4` |
| 2 | `convert_raw_rgb_to_thermal.py` | rgb tiles, `melobytes_medium_low_lut.npy` | `data/raw/.../thermal/*.mp4` |
| 3 | `build_frames_and_labels.py` | tiles, `data/annotations/raw_exports/project_2_rgb_annotations_adjusted.json` | `data/processed/{images,labels}/...` |
| 4 | `split_dataset_episodes.py` | tiles, labels | `data/splits/*` (repo-relative paths), `configs/data_{rgb,thermal}.yaml` |
| 5 | `train_full_detectors.py` | `configs/hyperparams.yaml` | `runs/detect/yolo11n_{rgb,thermal}_60e/` (`--smoke` for a 2-epoch check) |
| 6 | `extract_confidences.py` | weights, split lists | `runs/detect/unified_confidences_cache.json` |

After step 6 the cache changes only if the detectors change; steps 1 to 4 are deterministic (seed 0).

## Talk material (`talk/`)

| Script | Needs | Writes |
| :--- | :--- | :--- |
| `talk/blender_scene.py` | Blender 4.2+ | `figures/render_uav.png`: `blender -b --factory-startup -P scripts/talk/blender_scene.py -- figures/render_uav.png` |
| `talk/manim_window.py` | Manim 0.18+ | sliding-window animation: `manim -qh --media_dir scratch/manim -o window_filter scripts/talk/manim_window.py WindowFilter`, copy to `figures/anim_window_filter.mp4` |
| `talk/render_tikz.py` | pdflatex, pdftoppm | `figures/tikz_*.png`: every TikZ figure in `manuscript/` as a 300 dpi PNG, including five that are not in the paper |
