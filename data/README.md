# Data

| Path | In git | Contents |
| :--- | :---: | :--- |
| `splits/dataset_episodes_split.json` | yes | Parent-video split (Option B): 44 / 22 / 24 tiles, both tiles of a video in the same split, seed 0 |
| `splits/{train,val,test}_{rgb,thermal}.txt` | yes | Frame lists in feed order, repo-relative paths (`data/processed/images/...`) |
| `annotations/raw_exports/project_2_rgb_annotations_adjusted.json` | yes | Label Studio video export, the source of every label |
| `annotations/label_studio_config.xml` | yes | Label Studio labelling interface |
| `GENERATION_PROMPTS.md` | yes | Prompts used to generate the source videos |
| `raw/{biome}/{positive,negative}/{rgb,thermal}/*.mp4` | no (1.3 GB) | 90 tiles per stream, 640×640, 24 fps |
| `processed/` | no | Frames and YOLO labels written by `scripts/build_frames_and_labels.py` |
| `../videos/` | no | The 45 generated 1280×720 source videos |

The corpus: 45 generated videos (15 desert, 14 forest, 16 snow). Each video gives two 640×640 tiles
(left x = 0, right x = 640, y = 40), so 90 tiles per stream; 42 tiles contain a person and 48 do not
(31 videos show a person in at least one tile). The pseudo-thermal
stream is a 256-level colour lookup of RGB luminance (`scripts/melobytes_medium_low_lut.npy`); it carries no
temperature information.

All tables in the paper can be recomputed without any of the files marked "no": the per-frame detector
confidences are committed in `runs/detect/unified_confidences_cache.json`.
