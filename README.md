# SIFT from Scratch - Assignment 2

REPORT: https://www.overleaf.com/project/68f79836615fae8ddf252016
Complete implementation of SIFT (Scale-Invariant Feature Transform) algorithm from scratch in Python using NumPy and OpenCV for basic image operations.

## Features

- **Gaussian Pyramid**: Multi-scale representation with configurable octaves and scales
- **DoG Pyramid**: Difference-of-Gaussian for scale-space extrema detection
- **Keypoint Detection**: Scale-space extrema detection with 3D local maximum/minimum check
- **Keypoint Refinement**: Edge response filtering using Hessian matrix
- **Orientation Assignment**: 36-bin histogram for rotation invariance
- **128-D Descriptor**: 4×4 spatial grid × 8 orientation bins with normalization
- **Matching**: Brute-force matcher with Lowe's ratio test (0.75 threshold)
- **Evaluation**: Comprehensive tests for rotation, scale, and brightness robustness

## Project Structure

```
assignment_02_sift/
├── data/
│   ├── graffiti/           # Oxford dataset - viewpoint changes
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── bark/               # Oxford dataset - scale/rotation
│       ├── img1.jpg
│       └── img2.jpg
├── src/
│   ├── sift_from_scratch.py   # Complete SIFT implementation
│   ├── utils.py               # Visualization and helper functions
│   ├── evaluate.py            # Evaluation metrics and robustness tests
│   └── pairwise_match.py      # Pairwise matching across a folder of images
├── notebooks/
│   └── analysis.ipynb         # Interactive analysis and experiments
├── report/
│   ├── report.tex             # LaTeX report template
│   └── figures/               # Generated figures for report
├── README.md
└── run_demo.py                # Quick demo script
```

## Installation

```bash
pip install numpy opencv-python matplotlib
# Optional for OpenCV SIFT comparison:
pip install opencv-contrib-python
```

## Dataset

Download test images from the [Oxford Affine Covariant Regions Dataset](https://www.robots.ox.ac.uk/~vgg/research/affine/):

- **Graffiti**: [img1](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graffiti.jpg) (viewpoint changes)
- **Bark**: [img1](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bark.jpg) (scale/rotation)

Place downloaded images in `data/graffiti/` and `data/bark/` as `img1.jpg` and `img2.jpg`.

## Quick Start

### Run Demo

```bash
python run_demo.py
```

This will:
- Detect keypoints in image pairs
- Compute 128-D SIFT descriptors
- Match descriptors using ratio test
- Save visualizations to `report/figures/`

### Pairwise Image Matching (NEW!)

To find which images match in a collection:

```bash
# Place 5+ images in data/gallery/
python src/pairwise_match.py --folder data/gallery --out report/figures/gallery
```

This will:
- ✅ Compare all image pairs using SIFT keypoints
- ✅ Show which pairs match and **why** (number of matching keypoints)
- ✅ Generate match visualizations for each pair
- ✅ Create similarity matrix (CSV + heatmap)
- ✅ Save detailed matching report

**Output:**
- `matching_report.txt` - detailed explanation of which pairs match
- `matches_matrix.csv` - pairwise match counts
- `matches_heatmap.jpg` - visual similarity matrix
- `image1__image2_matches.jpg` - match visualizations for each pair

### Run Full Evaluation

```bash
python src/evaluate.py --data data --output report/figures
```

This runs comprehensive tests:
- Keypoint detection and matching on multiple pairs
- Rotation robustness (0°, 15°, 30°, 45°, 60°, 90°)
- Scale robustness (0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x)
- Brightness robustness (0.5x, 0.75x, 1.0x, 1.25x, 1.5x)
- Comparison with OpenCV SIFT (if available)

### Pairwise Gallery Matching (5 images or more)

Place multiple images in `data/gallery/` (jpg, png, ppm). Then run:

```powershell
python src/pairwise_match.py --folder data/gallery --out report/figures/gallery
```

This will:
- Detect keypoints and descriptors for each image
- Compute pairwise matches for all image pairs
- Save per-pair visualizations: `nameA__nameB_matches.jpg`
- Save a matches count matrix: `matches_matrix.csv`
- Save a heatmap visualization: `matches_heatmap.jpg`

### Interactive Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Usage Examples

### Basic SIFT Detection

```python
import cv2
from src.sift_from_scratch import SIFT_Scratch, match_descriptors

# Load images
img1 = cv2.imread('data/graffiti/img1.jpg')
img2 = cv2.imread('data/graffiti/img2.jpg')

# Detect and compute
sift = SIFT_Scratch(num_octaves=3, scales_per_octave=3)
kp1, desc1 = sift.detect_and_compute(img1)
kp2, desc2 = sift.detect_and_compute(img2)

# Match
matches = match_descriptors(desc1, desc2, ratio=0.75)
print(f"Matches: {len(matches)}")
```

### Customizing Parameters

```python
sift = SIFT_Scratch(
    num_octaves=4,          # Number of octaves in pyramid
    scales_per_octave=3,    # Scales per octave
    sigma=1.6,              # Initial Gaussian blur sigma
    contrast_thresh=0.04,   # Contrast threshold for extrema
    edge_thresh=10          # Edge response threshold
)
```

## Runtime Complexity

- **Gaussian Pyramid**: O(N × octaves × scales) where N = image pixels
- **DoG Pyramid**: O(N × octaves × scales)
- **Extrema Detection**: O(N × octaves × scales)
- **Descriptor Computation**: O(K × 256) where K = number of keypoints
- **Matching**: O(K₁ × K₂) for brute-force

## Report

Generate figures for your report:

1. Run evaluation: `python src/evaluate.py`
2. Open `notebooks/analysis.ipynb` for additional plots
3. Compile LaTeX report: `cd report && pdflatex report.tex`

## Performance Notes

This implementation prioritizes clarity and educational value. Key optimizations:

- Uses OpenCV's `GaussianBlur` for efficient convolution
- Vectorized numpy operations where possible
- Threshold-based candidate filtering before extrema check

For production use, consider:
- KD-tree or FLANN for faster matching (O(K log K))
- Full 3D quadratic interpolation for sub-pixel localization
- Parallel processing for pyramid construction

## References

1. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints". *International Journal of Computer Vision*, 60(2), 91-110.
2. [Oxford Affine Covariant Regions Dataset](https://www.robots.ox.ac.uk/~vgg/research/affine/)

## License

Educational use only.
