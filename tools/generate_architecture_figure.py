import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path

out_dir = Path('docs/manuscript/figures')
asset_dir = Path('docs/manuscript/figures/gen_assets')

# Figure dimensions: width 28 inches, height 12 inches (300 dpi)
fig, ax = plt.subplots(figsize=(28, 12), dpi=300)
ax.set_xlim(0, 28)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('white')

# Typography helpers
FONT_TITLE = 13.5
FONT_HEADER = 11.0
FONT_BODY = 8.5
FONT_SUB = 7.5

# -------------------------------------------------------------
# TOP CONTAINER: STAGE 1 - GENERATION & HARVESTING
# -------------------------------------------------------------
gen_box = patches.FancyBboxPatch((0.4, 6.1), 27.2, 5.5, boxstyle='round,pad=0.2,rounding_size=0.35',
                                linewidth=2.2, edgecolor='#1D4ED8', facecolor='#F0F7FF', zorder=1)
ax.add_patch(gen_box)

ax.text(0.8, 11.2, 'STAGE 1: GENERATION (MULTIMODAL SYNTHETIC GENERATION & 2× DUAL-CROP HARVESTING)',
        fontsize=FONT_TITLE, fontweight='bold', color='#1E40AF', zorder=2)

# Load Icons
lorem_img = Image.open(asset_dir / 'icon_lorem_ipsum.png')
gemini_img = Image.open(asset_dir / 'icon_gemini.png')

# 1. RGB video prompt
x_c1 = 1.8
ax.text(x_c1, 10.5, 'RGB video prompt', fontsize=FONT_HEADER, fontweight='bold', color='#111827', ha='center', zorder=3)
ax.imshow(lorem_img, extent=(x_c1 - 0.7, x_c1 + 0.7, 8.1, 10.0), zorder=3)
ax.text(x_c1, 7.6, 'Text Description\n(Prompt for Desert & Forest)', fontsize=FONT_BODY, color='#4B5563', ha='center', zorder=3)

# Arrow 1 -> 2
ax.annotate('', xy=(3.4, 9.05), xytext=(2.7, 9.05),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#2563EB'), zorder=4)

# 2. Video Generator LLM (Gemini)
x_c2 = 4.3
ax.text(x_c2, 10.5, 'Video Generator LLM', fontsize=FONT_HEADER, fontweight='bold', color='#4338CA', ha='center', zorder=3)
ax.imshow(gemini_img, extent=(x_c2 - 0.75, x_c2 + 0.75, 8.3, 9.8), zorder=3)
ax.text(x_c2, 7.6, 'Gemini 3.8 Flash\n(Google DeepMind)', fontsize=FONT_BODY, color='#4B5563', ha='center', zorder=3)

# Arrow 2 -> 3
ax.annotate('', xy=(5.8, 9.05), xytext=(5.1, 9.05),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#2563EB'), zorder=4)

# 3. 16:9 RGB Videos (Desert & Forest with Striped Bottom)
x_c3 = 7.7
ax.text(x_c3, 10.5, '16:9 RGB Videos', fontsize=FONT_HEADER, fontweight='bold', color='#111827', ha='center', zorder=3)
d_rgb_v = Image.open(asset_dir / 'desert_rgb_video.png')
f_rgb_v = Image.open(asset_dir / 'forest_rgb_video.png')
ax.imshow(d_rgb_v, extent=(6.2, 7.55, 8.35, 9.85), zorder=3)
ax.text(6.87, 8.05, 'Desert RGB', fontsize=8, fontweight='bold', color='#1F2937', ha='center', zorder=3)
ax.imshow(f_rgb_v, extent=(7.85, 9.2, 8.35, 9.85), zorder=3)
ax.text(8.52, 8.05, 'Forest RGB', fontsize=8, fontweight='bold', color='#1F2937', ha='center', zorder=3)
ax.text(x_c3, 7.45, '1280×720 @ 24 FPS (10.0 s)\n[Diagonal Striped Bottom = Video]', fontsize=7.8, color='#4B5563', ha='center', zorder=3)

# Arrow 3 -> 4
ax.annotate('', xy=(10.0, 9.05), xytext=(9.3, 9.05),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#D97706'), zorder=4)

# 4. Configure to Thermal
x_c4 = 11.0
ax.text(x_c4, 10.5, 'Configure to Thermal', fontsize=FONT_HEADER, fontweight='bold', color='#B45309', ha='center', zorder=3)
ax.imshow(lorem_img, extent=(x_c4 - 0.7, x_c4 + 0.7, 8.1, 10.0), zorder=3)
ax.text(x_c4, 7.6, 'Thermal Modality Prompt\n(LWIR Sensor Radiance)', fontsize=FONT_BODY, color='#4B5563', ha='center', zorder=3)

# Arrow 4 -> 5
ax.annotate('', xy=(12.6, 9.05), xytext=(11.9, 9.05),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#D97706'), zorder=4)

# 5. Video Generator LLM (Thermal Translation)
x_c5 = 13.5
ax.text(x_c5, 10.5, 'Video Generator LLM', fontsize=FONT_HEADER, fontweight='bold', color='#4338CA', ha='center', zorder=3)
ax.imshow(gemini_img, extent=(x_c5 - 0.75, x_c5 + 0.75, 8.3, 9.8), zorder=3)
ax.text(x_c5, 7.6, 'Modality Translation\n(Gemini Video-to-Video)', fontsize=FONT_BODY, color='#4B5563', ha='center', zorder=3)

# Arrow 5 -> 6
ax.annotate('', xy=(15.0, 9.05), xytext=(14.3, 9.05),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#D97706'), zorder=4)

# 6. 16:9 Thermal Videos (Desert & Forest with Striped Bottom)
x_c6 = 16.9
ax.text(x_c6, 10.5, '16:9 Thermal Videos', fontsize=FONT_HEADER, fontweight='bold', color='#111827', ha='center', zorder=3)
d_th_v = Image.open(asset_dir / 'desert_th_video.png')
f_th_v = Image.open(asset_dir / 'forest_th_video.png')
ax.imshow(d_th_v, extent=(15.4, 16.75, 8.35, 9.85), zorder=3)
ax.text(16.07, 8.05, 'Desert Thermal (D:)', fontsize=8, fontweight='bold', color='#1F2937', ha='center', zorder=3)
ax.imshow(f_th_v, extent=(17.05, 18.4, 8.35, 9.85), zorder=3)
ax.text(17.72, 8.05, 'Forest Thermal', fontsize=8, fontweight='bold', color='#1F2937', ha='center', zorder=3)
ax.text(x_c6, 7.45, '1280×720 @ 24 FPS (10.0 s)\n[Diagonal Striped Bottom = Video]', fontsize=7.8, color='#4B5563', ha='center', zorder=3)

# Arrow 6 -> 7
ax.annotate('', xy=(19.1, 9.05), xytext=(18.5, 9.05),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#059669'), zorder=4)

# 7. Video to Frame
x_c7 = 20.0
v2f_box = patches.FancyBboxPatch((x_c7 - 0.8, 8.1), 1.6, 1.9, boxstyle='round,pad=0.08,rounding_size=0.2',
                                 linewidth=1.8, edgecolor='#059669', facecolor='#ECFDF5', zorder=3)
ax.add_patch(v2f_box)
ax.text(x_c7, 9.35, 'Video to\nFrame', fontsize=10.5, fontweight='bold', color='#065F46', ha='center', va='center', zorder=4)
ax.text(x_c7, 8.45, 'Frame Extraction\n3:1 Decimate\n(24 → 8 Hz)', fontsize=7.2, color='#047857', ha='center', va='center', zorder=4)

# Arrow 7 -> 8
ax.annotate('', xy=(21.5, 9.05), xytext=(20.9, 9.05),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#059669'), zorder=4)

# 8. 16:9 Clean Frames (NO stripes)
x_c8 = 22.9
ax.text(x_c8, 10.5, '16:9 Frames (Extracted)', fontsize=FONT_HEADER, fontweight='bold', color='#111827', ha='center', zorder=3)
d_rgb_f = Image.open(asset_dir / 'desert_rgb_frame.png')
d_th_f = Image.open(asset_dir / 'desert_th_frame.png')
ax.imshow(d_rgb_f, extent=(21.7, 22.8, 8.4, 9.8), zorder=3)
ax.text(22.25, 8.1, 'RGB Frame', fontsize=7.5, fontweight='bold', color='#1F2937', ha='center', zorder=3)
ax.imshow(d_th_f, extent=(23.0, 24.1, 8.4, 9.8), zorder=3)
ax.text(23.55, 8.1, 'Thermal Frame', fontsize=7.5, fontweight='bold', color='#1F2937', ha='center', zorder=3)
ax.text(x_c8, 7.5, '1280×720 (No Stripes)', fontsize=7.8, color='#4B5563', ha='center', zorder=3)

# Arrow 8 -> 9
ax.annotate('', xy=(24.8, 9.05), xytext=(24.2, 9.05),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#7C3AED'), zorder=4)

# 9. Dual Crop 640x640 (2 of them: Left & Right)
x_c9 = 26.2
ax.text(x_c9, 10.5, 'Crop 640×640 (2 of them)', fontsize=FONT_HEADER, fontweight='bold', color='#6D28D9', ha='center', zorder=3)
d_rgb_l = Image.open(asset_dir / 'desert_rgb_crop_left.png')
d_rgb_r = Image.open(asset_dir / 'desert_rgb_crop_right.png')
ax.imshow(d_rgb_l, extent=(25.0, 26.05, 8.4, 9.8), zorder=3)
ax.text(25.52, 8.1, 'Left: x∈[0, 640]', fontsize=7.2, fontweight='bold', color='#1F2937', ha='center', zorder=3)
ax.imshow(d_rgb_r, extent=(26.35, 27.4, 8.4, 9.8), zorder=3)
ax.text(26.87, 8.1, 'Right: x∈[640, 1280]', fontsize=7.2, fontweight='bold', color='#1F2937', ha='center', zorder=3)
ax.text(x_c9, 7.45, 'Actual Dataset Frames\n(2× Dataset Size, y=40)', fontsize=7.8, fontweight='bold', color='#6D28D9', ha='center', zorder=3)

# -------------------------------------------------------------
# BOTTOM CONTAINER: STAGE 2 - DETECTION, FUSION & POST-PROCESSING
# -------------------------------------------------------------
mmsar_box = patches.FancyBboxPatch((0.4, 0.4), 27.2, 5.2, boxstyle='round,pad=0.2,rounding_size=0.35',
                                  linewidth=2.2, edgecolor='#059669', facecolor='#F0FDF4', zorder=1)
ax.add_patch(mmsar_box)

ax.text(0.8, 5.15, 'STAGE 2: ONBOARD MULTIMODAL DETECTION, DECISION FUSION & CAUSAL TEMPORAL POST-PROCESSING',
        fontsize=FONT_TITLE, fontweight='bold', color='#047857', zorder=2)

# Connector Bar between Stage 1 and Stage 2
ax.plot([26.2, 27.3, 27.3, 1.4, 1.4, 2.2], [7.3, 7.3, 5.75, 5.75, 3.5, 3.5],
        lw=2.8, color='#7C3AED', linestyle='--', zorder=5)
ax.annotate('', xy=(2.4, 3.5), xytext=(1.4, 3.5),
            arrowprops=dict(arrowstyle='->', lw=2.8, color='#7C3AED'), zorder=5)

ax.text(14.0, 5.75, 'Harvested Dataset: 30 Snippets per Environment (60% Train / 20% Val / 20% Test) — 80 Frames/Clip @ 8 Hz',
        fontsize=10.0, fontweight='bold', color='#4C1D95', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#EDE9FE', edgecolor='#8B5CF6', lw=1.6), zorder=6)

# Stage 2 Blocks
# 1. Detectors
det_rgb = patches.FancyBboxPatch((2.4, 3.6), 3.8, 1.15, boxstyle='round,pad=0.1,rounding_size=0.2',
                                linewidth=1.8, edgecolor='#0284C7', facecolor='#E0F2FE', zorder=3)
ax.add_patch(det_rgb)
ax.text(4.3, 4.3, 'YOLO11n - RGB Detector', fontsize=10.5, fontweight='bold', color='#0369A1', ha='center', zorder=4)
ax.text(4.3, 3.85, '2.6M Parameters | FP32 | Batch 16', fontsize=8.0, color='#0284C7', ha='center', zorder=4)

det_th = patches.FancyBboxPatch((2.4, 1.4), 3.8, 1.15, boxstyle='round,pad=0.1,rounding_size=0.2',
                               linewidth=1.8, edgecolor='#D97706', facecolor='#FEF3C7', zorder=3)
ax.add_patch(det_th)
ax.text(4.3, 2.1, 'YOLO11n - Thermal Detector', fontsize=10.5, fontweight='bold', color='#B45309', ha='center', zorder=4)
ax.text(4.3, 1.65, '2.6M Parameters | FP32 | Batch 16', fontsize=8.0, color='#D97706', ha='center', zorder=4)

# Inputs to Detectors
ax.text(1.2, 4.15, 'RGB Stream\n640×640 (8 Hz)', fontsize=7.8, fontweight='bold', color='#0369A1', ha='center', zorder=4)
ax.annotate('', xy=(2.4, 4.15), xytext=(1.7, 4.15), arrowprops=dict(arrowstyle='->', lw=2.0, color='#0284C7'), zorder=4)

ax.text(1.2, 1.95, 'Thermal Stream\n640×640 (8 Hz)', fontsize=7.8, fontweight='bold', color='#B45309', ha='center', zorder=4)
ax.annotate('', xy=(2.4, 1.95), xytext=(1.7, 1.95), arrowprops=dict(arrowstyle='->', lw=2.0, color='#D97706'), zorder=4)

# Arrows from Detectors to Fusion
ax.annotate('', xy=(7.8, 3.2), xytext=(6.2, 4.15), arrowprops=dict(arrowstyle='->', lw=2.0, color='#0284C7'), zorder=4)
ax.text(6.8, 3.9, 'c_RGB[n]', fontsize=9.5, fontweight='bold', color='#0369A1', zorder=4)

ax.annotate('', xy=(7.8, 2.8), xytext=(6.2, 1.95), arrowprops=dict(arrowstyle='->', lw=2.0, color='#D97706'), zorder=4)
ax.text(6.8, 2.2, 'c_Thermal[n]', fontsize=9.5, fontweight='bold', color='#B45309', zorder=4)

# Decision Late Fusion Gate
fusion_box = patches.FancyBboxPatch((7.8, 2.2), 3.4, 1.5, boxstyle='round,pad=0.1,rounding_size=0.2',
                                   linewidth=2.0, edgecolor='#7C3AED', facecolor='#F3E8FF', zorder=3)
ax.add_patch(fusion_box)
ax.text(9.5, 3.2, 'Decision Late Fusion', fontsize=10.5, fontweight='bold', color='#6D28D9', ha='center', zorder=4)
ax.text(9.5, 2.8, 's[n] = max(c_RGB, c_Thermal)', fontsize=9.5, fontweight='bold', color='#4C1D95', ha='center', zorder=4)
ax.text(9.5, 2.45, 'Soft Disjunctive Max-Pool', fontsize=8.0, color='#7C3AED', ha='center', zorder=4)

# Distribution bus
ax.plot([11.2, 12.3], [2.95, 2.95], lw=2.5, color='#7C3AED', zorder=4)
ax.text(11.75, 3.2, 's[n]', fontsize=10, fontweight='bold', color='#7C3AED', ha='center', zorder=4)
ax.plot([12.3, 12.3], [1.0, 4.8], lw=2.5, color='#7C3AED', zorder=4)

# 5 Post-Processing Methods
methods = [
    ('M1: Baseline Raw Thresholding', 'y[n] = 1(s[n] >= tau_raw*)', 4.8, '#DC2626', '#FEF2F2'),
    ('M2: Five-Frame Moving Average', 's_MA[n] = (1/W) * sum(s[n-k])', 3.85, '#D97706', '#FFFBEB'),
    ('M3: Five-Frame Median Filter', 's_med[n] = median(s[n], ..., s[n-4])', 2.9, '#059669', '#ECFDF5'),
    ('M4: History Consensus (3-of-5)', 'sum(1(s[n-k] >= tau*)) >= 3', 1.95, '#2563EB', '#EFF6FF'),
    ('M5: Learned Mamba-SSSM', 'Selective State-Space Model (d_state=8, O(1))', 1.0, '#7C3AED', '#F5F3FF')
]

for name, formula, y_pos, border_col, bg_col in methods:
    ax.annotate('', xy=(12.9, y_pos), xytext=(12.3, y_pos),
                arrowprops=dict(arrowstyle='->', lw=2.0, color='#7C3AED'), zorder=4)
    m_box = patches.FancyBboxPatch((12.9, y_pos - 0.35), 4.7, 0.7, boxstyle='round,pad=0.08,rounding_size=0.15',
                                  linewidth=1.5, edgecolor=border_col, facecolor=bg_col, zorder=3)
    ax.add_patch(m_box)
    ax.text(15.25, y_pos + 0.1, name, fontsize=8.8, fontweight='bold', color=border_col, ha='center', zorder=4)
    ax.text(15.25, y_pos - 0.18, formula, fontsize=7.6, color='#374151', ha='center', zorder=4)
    ax.plot([17.6, 18.4], [y_pos, y_pos], lw=2.0, color='#0D9488', zorder=4)

# Collection Bus to Alarm
ax.plot([18.4, 18.4], [1.0, 4.8], lw=2.5, color='#0D9488', zorder=4)
ax.annotate('', xy=(19.1, 2.9), xytext=(18.4, 2.9),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#0D9488'), zorder=4)

# Binary Alarm Output
alarm_box = patches.FancyBboxPatch((19.1, 2.15), 2.7, 1.5, boxstyle='round,pad=0.1,rounding_size=0.2',
                                  linewidth=2.0, edgecolor='#0D9488', facecolor='#CCFBF1', zorder=3)
ax.add_patch(alarm_box)
ax.text(20.45, 3.15, 'Binary Alarm', fontsize=11, fontweight='bold', color='#0F766E', ha='center', zorder=4)
ax.text(20.45, 2.75, 'y_m[n] in {0, 1}', fontsize=10, fontweight='bold', color='#115E59', ha='center', zorder=4)
ax.text(20.45, 2.4, 'Optimal Val tau_m*', fontsize=8.2, color='#0D9488', ha='center', zorder=4)

# Arrow from Alarm to Evaluation
ax.annotate('', xy=(22.6, 2.9), xytext=(21.8, 2.9),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#0D9488'), zorder=4)

# Operational Evaluation Framework
eval_box = patches.FancyBboxPatch((22.6, 0.8), 4.8, 2.7, boxstyle='round,pad=0.12,rounding_size=0.25',
                                 linewidth=2.0, edgecolor='#16A34A', facecolor='#DCFCE7', zorder=3)
ax.add_patch(eval_box)
ax.text(25.0, 3.1, 'Operational Evaluation Framework', fontsize=11.0, fontweight='bold', color='#15803D', ha='center', zorder=4)
eval_points = [
    '• Upstream Detection (Precision, Recall, F1)',
    '• False Alarm Suppression Ratio (FASR)',
    '• Satellite / Cellular IoT Telemetry Savings (ΔΩ)',
    '• Autonomous UAV Hover Battery Energy Drain (ΔE)'
]
for idx, pt in enumerate(eval_points):
    ax.text(22.8, 2.65 - idx * 0.42, pt, fontsize=8.0, color='#166534', zorder=4)

# Top KPI callout in Evaluation
kpi_box = patches.FancyBboxPatch((22.6, 3.8), 4.8, 1.1, boxstyle='round,pad=0.08,rounding_size=0.15',
                                linewidth=1.6, edgecolor='#DC2626', facecolor='#FEF2F2', zorder=3)
ax.add_patch(kpi_box)
ax.text(25.0, 4.45, 'False Alarm Suppression Ratio (FASR)', fontsize=9.0, fontweight='bold', color='#991B1B', ha='center', zorder=4)
ax.text(25.0, 4.05, 'FASR = [ 1 - ( FP_filtered / FP_raw ) ] * 100%',
        fontsize=9.0, fontweight='bold', color='#B91C1C', ha='center', zorder=4)

out_img_path = out_dir / 'system_architecture_extended.png'
fig.savefig(out_img_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f'Successfully generated: {out_img_path}')
