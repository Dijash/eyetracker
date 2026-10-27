# EyeTracker 👁️

> **AI-Powered Visual Attention Prediction, Heatmap Overlay & Social Media Viewability Engine**

EyeTracker is a Python engine that predicts where human eyes look when viewing an image or creative design. It calculates biological visual saliency, generates smooth eye-tracking heatmap overlays, extracts the sequential saccadic scanpath (1st, 2nd, 3rd fixations), and evaluates **estimated views, Click-Through Rates (CTR), and safe-zone compliance** across major social media platforms.

---

## 🚀 Key Features

- **Biological Visual Attention Engine**:
  - **Fourier Spectral Residual Novelty** (Xiaodi Hou & Lingqing Zhang, CVPR) detects pop-out focal subjects, anomalies, and isolated graphical hooks.
  - **Retinal Color Opponency** ($RG$ Red-Green and $BY$ Blue-Yellow) with biological warmth/skin-tone attraction bias.
  - **Multi-Scale Spatial Gradients** (fine and coarse Sobel energy) capturing text density, facial features, and high-contrast boundaries.
  - **Human Fixation Priors**: Anisotropic 2D Gaussian screen center bias and Rule-of-Thirds nodal attraction.
  - **Foveal Gaussian Smoothing**: Models the human foveal receptive field (~2° of visual field).
- **Saccadic Scanpath Extractor**:
  - Automatically extracts the numbered sequence of visual fixations (**#1 Focal Hook**, **#2 Secondary Interest**, **#3 Context**, etc.) using Gaussian Inhibition of Return (Non-Maximum Suppression).
- **Eye-Tracking Heatmap Overlays**:
  - Smooth alpha-blended rendering onto the original image. Low-saliency areas remain clear, while hot areas glow smoothly.
  - Built-in vectorized colormaps: `turbo` (Google Turbo, default), `jet`, `inferno`, and `hot`.
  - Zero heavy framework dependencies (runs at blazing speed in pure NumPy + Pillow).
- **Social Media Viewability & Engagement Calculator**:
  - Analyzes 8 major social media formats:
    - **YouTube** (16:9 Thumbnail) with **Duration Timestamp Collision Detection**
    - **Instagram Feed** (1:1 Square & 4:5 Portrait)
    - **TikTok** (9:16 Full-Screen Video Cover) with **Right-Side Button & Bottom Caption Safe-Zone Auditing**
    - **Instagram Reels / Stories** (9:16 Vertical)
    - **X / Twitter** (16:9 In-Feed Card)
    - **LinkedIn** (1.91:1 Landscape & 1:1 Feed Post)
    - **Pinterest** (2:3 Vertical Pin)
  - Computes:
    - **Gaze Retention %**: Proportion of visual attention preserved within the platform's viewport crop.
    - **UI Safe-Zone Compliance %**: Flags when important text/logos/faces are covered by platform badges or icons.
    - **Estimated Click-Through Rate (CTR %)**: Realistic platform-calibrated CTR estimation.
    - **Thumb-Stop Hook Rate %**: Probability that a fast-scrolling user pauses on the post.
    - **Platform Viewability Score (0-100)** & Grade (`S`, `A+`, `A`, `B`, `C`, `D`).
    - **Cross-Platform Relative View Multiplier**: Ranks where your creative will perform best.
- **Multi-Platform Visual Simulation Grid**:
  - Automatically exports a side-by-side composite audit card showing original image, attention heatmap, scanpath, and real-world crop simulations for YouTube, Instagram, and TikTok with UI overlays drawn.

---

## 📦 Installation & Requirements

EyeTracker requires **Python 3.8+** with `numpy` and `pillow`. `rich` is optional for styled terminal output.

```bash
pip install numpy pillow rich
```

*(No PyTorch or Matplotlib required — self-contained, lightweight, and fast!)*

---

## ⚡ Quickstart (CLI)

### 1. Run Demo on Sample Creative
Test EyeTracker with a built-in realistic YouTube thumbnail creative:

```bash
python3 eyetracker.py --demo
```

### 2. Analyze Your Own Image
```bash
python3 eyetracker.py --image path/to/your_image.jpg --output-dir ./results
```

### 3. Customize Options
```bash
python3 eyetracker.py \
  --image thumbnail.png \
  --output-dir ./audit_output \
  --colormap turbo \
  --alpha 0.70 \
  --points 5 \
  --json
```

---

## 📊 Sample Terminal Output

```
╭───────────────────── EyeTracker Visual Attention Audit ──────────────────────╮
│ Input Dimensions: 1280x720 (Aspect: 1.778)                                   │
│ Focal Hook Dominance: 69.9/100 | Clutter Level: 41.1/100                     │
│ Best Platform Fit: YouTube (Video Thumbnail)                                 │
│ Highest Risk Fit: TikTok (Full-Screen Video / Cover)                         │
╰──────────────────────────────────────────────────────────────────────────────╯
          Human Fixation Saccade Sequence (Order of Attention)           
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Order ┃ Coordinates (X, Y) ┃ Attention Energy ┃ Local Attention Share ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│  #1   │     (652, 363)     │       100%       │         25.8%         │
│  #2   │     (831, 228)     │       82%        │         19.1%         │
│  #3   │     (431, 299)     │       80%        │         18.4%         │
│  #4   │    (1051, 226)     │       68%        │         8.0%          │
│  #5   │     (787, 408)     │       98%        │         20.9%         │
└───────┴────────────────────┴──────────────────┴───────────────────────┘
           Social Media Platform Viewability & Approx Views Analysis            
┏━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┓
┃ Plat… ┃       ┃  Gaze ┃    UI ┃  Est. ┃  Hook ┃       ┃       ┃ View  ┃ Not… ┃
┃ /     ┃ Aspe… ┃  Kept ┃  Safe ┃   CTR ┃ Stop% ┃ Score ┃ Grade ┃ Mult… ┃ War… ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━┩
│ YouT… │ 16:9  │ 100.% │ 100.% │ 5.90% │ 90.1% │  92.4 │   S   │ 1.12x │ Opt… │
│ Inst… │  1:1  │ 83.3% │ 100.% │ 7.27% │ 85.7% │  85.7 │  A+   │ 1.04x │ Opt… │
│ Inst… │  4:5  │ 71.5% │ 100.% │ 8.60% │ 82.7% │  81.0 │   A   │ 0.99x │ Opt… │
│ TikT… │ 9:16  │ 54.6% │ 72.2% │ 7.64% │ 72.8% │  65.9 │   B   │ 0.80x │ Act… │
│ Inst… │ 9:16  │ 54.6% │ 90.6% │ 7.15% │ 76.4% │  71.4 │   B   │ 0.87x │ Top… │
│ X     │ 16:9  │ 100.% │ 100.% │ 5.33% │ 90.1% │  92.4 │   S   │ 1.12x │ Opt… │
│ Link… │ 1.91… │ 97.5% │ 100.% │ 4.73% │ 89.4% │  91.4 │   S   │ 1.11x │ Opt… │
│ Pint… │  2:3  │ 62.5% │ 100.% │ 7.09% │ 80.3% │  77.4 │   A   │ 0.94x │ Opt… │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴──────┘
```

---

## 🎨 Generated Output Files

Running EyeTracker produces five key files in your output directory:

| Filename | Description |
| :--- | :--- |
| `[name]_heatmap_overlay.png` | Original image with color heatmap overlay and gradient colorbar legend. |
| `[name]_scanpath.png` | Numbered fixation points (#1, #2, #3...) with saccadic connecting scanpath arrows. |
| `[name]_saliency_raw.png` | Raw 8-bit grayscale saliency probability map. |
| `[name]_social_audit_grid.png` | Composite visual audit board with side-by-side YouTube, Instagram, and TikTok crops. |
| `[name]_report.json` | Complete machine-readable JSON metrics containing all coordinates, scores, and audits. |

---

## 💻 Python API Usage

You can easily embed EyeTracker into your Python workflows, web backends, or thumbnail generation pipelines:

```python
from PIL import Image
from eyetracker import EyeTracker

# Initialize tracker
tracker = EyeTracker()

# Analyze an image (accepts file path, PIL Image, or NumPy array)
result = tracker.analyze("my_creative.jpg", num_fixation_points=5)

# 1. Inspect Fixations
for fix in result.fixation_points:
    print(f"Fixation #{fix.order}: ({fix.x}, {fix.y}) - Attention Share: {fix.attention_share_pct}%")

# 2. Inspect Social Media Audits
for audit in result.platform_audits:
    print(f"{audit.platform_name} ({audit.format_label}):")
    print(f"  • Viewability Score: {audit.viewability_score}/100 [Grade {audit.grade}]")
    print(f"  • Estimated CTR:     {audit.estimated_ctr_pct}%")
    print(f"  • Gaze Retention:    {audit.gaze_retention_pct}%")
    print(f"  • UI Safe Zone:      {audit.safe_zone_pct}%")
    if audit.danger_zone_collisions:
        print(f"  ⚠️ Warning:          {audit.danger_zone_collisions}")

# 3. Save all visualization artifacts
saved = tracker.render_all_outputs(
    orig_img=Image.open("my_creative.jpg"),
    result=result,
    output_dir="./output_folder",
    colormap="turbo",
    alpha=0.65,
)
print("Saved files:", saved)
```

---

## 🔬 Scientific Methodology

### 1. Human Visual Attention Saliency Model
EyeTracker models both bottom-up preattentive mechanisms and top-down ecological viewing biases:

1. **Spectral Residual (Fourier Domain Novelty)**:
   Natural images follow an average $1/f$ log-amplitude spectrum $\mathcal{L}(f)$. Saliency corresponds to statistically unexpected anomalies:
   $$\mathcal{R}(f) = \mathcal{L}(f) - h_n(f) * \mathcal{L}(f)$$
   Reconstruction via inverse 2D FFT highlights pop-out objects, logos, and high-contrast headlines.
2. **Color Opponency & Chroma Warmth**:
   Retinal ganglion and LGN pathways process signals along red-green ($R - G$) and blue-yellow ($B - \frac{R+G}{2}$) axes. Biological bias weights warm hues (skin tones, red CTA buttons) more heavily than cool backgrounds.
3. **Multi-Scale Spatial Gradient Energy**:
   Multi-scale Sobel gradients isolate sharp text glyphs, facial eyes/lips, and silhouette boundaries.
4. **Anisotropic Center & Golden Ratio Priors**:
   Eye-tracking benchmarks (MIT Saliency Benchmark, SALICON) demonstrate that human gazes naturally cluster near the image center and along Rule-of-Thirds intersections.

### 2. Saccadic Scanpath Order
Humans do not view images simultaneously; eyes perform rapid ballistics (**saccades**) between discrete fixation pauses:
- The global maximum is Fixation #1 (the primary hook).
- Gaussian Inhibition of Return temporarily suppresses the immediate neighborhood around Fixation #1, allowing the model to find Fixation #2 (secondary focal element), Fixation #3, etc.

### 3. Platform Viewability & Estimated CTR Formula
Each platform is evaluated across 4 core factors:
- **Gaze Retention ($R$)**: Fraction of total saliency inside the aspect ratio crop.
- **Safe-Zone Integrity ($S$)**: Fraction of gaze outside platform UI danger zones (e.g., YouTube duration badge $X \ge 78\%, Y \ge 76\%$, TikTok right sidebar $X \ge 82\%$).
- **Visual Clarity ($C$)**: Low-clutter, dominant-subject bonus.
- **Composition ($P$)**: Rule-of-Thirds alignment.

The composite Viewability Index $V \in [0, 100]$ predicts performance, scaling platform-calibrated baseline Click-Through Rates non-linearly:
$$\text{CTR} = \text{Base} \times \left(0.55 + 0.95 \times \left(\frac{V}{100}\right)^{1.35}\right)$$

---

## 🧪 Testing

Run the test suite:

```bash
python3 test_eyetracker.py
```

