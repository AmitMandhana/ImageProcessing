# SIFT from Scratch - Quick Reference Card

## 📁 Project Structure
```
assignment_02_sift/
├── src/
│   ├── sift_from_scratch.py    ← Main SIFT implementation
│   ├── utils.py                ← Helper functions
│   └── evaluate.py             ← Evaluation scripts
├── data/
│   ├── graffiti/               ← Place images here
│   └── bark/                   ← Place images here
├── notebooks/
│   └── analysis.ipynb          ← Interactive experiments
├── report/
│   ├── report.tex              ← LaTeX report
│   └── figures/                ← Generated figures
├── run_demo.py                 ← Quick demo script
├── requirements.txt            ← Dependencies
├── README.md                   ← Full documentation
├── SETUP_INSTRUCTIONS.md       ← Detailed setup guide
└── .gitignore                  ← Git ignore rules
```

## ⚡ Quick Commands

### Setup
```powershell
# Install dependencies
pip install -r requirements.txt

# Or manually
pip install numpy opencv-python matplotlib
```

### Run
```powershell
# Quick demo
python run_demo.py

# Full evaluation
python src/evaluate.py

# Direct test on images
python src/sift_from_scratch.py data/graffiti/img1.jpg data/graffiti/img2.jpg

# Interactive analysis
jupyter notebook notebooks/analysis.ipynb
```

## 🎯 Key Classes & Functions

### Main SIFT Class
```python
from src.sift_from_scratch import SIFT_Scratch

sift = SIFT_Scratch(
    num_octaves=3,           # Scale pyramid levels
    scales_per_octave=3,     # Scales per level
    contrast_thresh=0.04,    # Min contrast
    edge_thresh=10           # Max edge ratio
)

kp, desc = sift.detect_and_compute(image)
```

### Matching
```python
from src.sift_from_scratch import match_descriptors

matches = match_descriptors(desc1, desc2, ratio=0.75)
```

### Visualization
```python
from src.sift_from_scratch import draw_matches

img_out = draw_matches(img1, kp1, img2, kp2, matches)
```

## 📊 Dataset Links

**Oxford Affine Covariant Regions:**
- Main page: https://www.robots.ox.ac.uk/~vgg/research/affine/
- Graffiti: https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graffiti.jpg
- Bark: https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bark.jpg

**Save images as:**
- `data/graffiti/img1.jpg` and `img2.jpg`
- `data/bark/img1.jpg` and `img2.jpg`

## 🔧 Common Issues

| Problem | Solution |
|---------|----------|
| No keypoints detected | Lower `contrast_thresh` to 0.02 |
| Too slow | Reduce image size or use fewer octaves |
| Import errors | Run `pip install -r requirements.txt` |
| Images not found | Check paths in `data/` folders |

## 📈 Expected Output

**Console output:**
```
Processing pair: img1.jpg vs img2.jpg
Detecting keypoints and computing descriptors...
Keypoints detected: 342 (img1), 289 (img2)
Matching descriptors...
Matches found: 127
Saved visualization to: report/figures/graffiti_matches.jpg
```

**Generated files:**
- `report/figures/keypoints_img1.jpg`
- `report/figures/keypoints_img2.jpg`
- `report/figures/matches.jpg`
- `report/figures/rotation_curve.png`
- `report/figures/scale_curve.png`
- `report/figures/brightness_curve.png`
- `report/figures/results_summary.txt`

## 🎓 For Your Report

**Include these sections:**
1. Introduction (SIFT overview)
2. Implementation (algorithm details)
3. Experiments (your results with figures)
4. Discussion (analysis & comparison)
5. Conclusion

**Include these figures:**
- Keypoint detection visualization
- Match visualization (graffiti + bark)
- Rotation robustness curve
- Scale robustness curve
- Comparison table (custom vs OpenCV)

**Key metrics to report:**
- Keypoints detected: ~300-500 per image
- Matches: ~100-200 per pair
- Runtime: ~2-3 seconds (custom) vs ~0.2s (OpenCV)
- Rotation invariance: stable up to 60°
- Scale invariance: 0.5× to 2×

## 💡 Bonus Features to Add

```python
# Homography estimation + warping
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
warped = cv2.warpPerspective(img2, H, (w, h))

# Descriptor statistics
print(f"Descriptor mean: {desc.mean():.4f}")
print(f"Descriptor std: {desc.std():.4f}")

# Match quality (if ground truth available)
inliers = mask.sum()
precision = inliers / len(matches)
```

## 🚀 Optimization Ideas

**For faster runtime:**
- Reduce `num_octaves` to 3
- Use smaller images (resize to 640px width)
- Implement KD-tree matching instead of brute-force

**For better quality:**
- Implement 3D quadratic interpolation
- Add Gaussian weighting to descriptors
- Use FLANN matcher

## 📝 Citation

```latex
@article{lowe2004distinctive,
  title={Distinctive image features from scale-invariant keypoints},
  author={Lowe, David G},
  journal={International journal of computer vision},
  volume={60},
  number={2},
  pages={91--110},
  year={2004}
}
```

---
**Good luck! 🎯**
