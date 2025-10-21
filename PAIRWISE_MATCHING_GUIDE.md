# Pairwise Image Matching - Quick Guide

## What It Does

The pairwise matching script compares **all pairs** of images in a folder and tells you:
- ✅ **Which pairs match** (show the same scene/object)
- ✅ **Why they match** (based on number of matching SIFT keypoints)
- ✅ **How well they match** (match quality percentage)

## Usage

### Step 1: Add Images

Place 5 or more images in the `data/gallery/` folder:

```
data/gallery/
  ├── photo1.jpg
  ├── photo2.jpg
  ├── photo3.jpg
  ├── photo4.jpg
  └── photo5.jpg
```

**Tip:** For best results, include:
- 2-3 images of the same scene from different angles (these should match)
- 2-3 images of completely different scenes (these should NOT match)

### Step 2: Run Pairwise Matching

```powershell
python src/pairwise_match.py --folder data/gallery --out report/figures/gallery
```

### Step 3: Check Results

The script will print to console:

```
graf1 <-> graf2
  Keypoints: 495 vs 537
  Matches found: 128
  Status: ✓ MATCHING
  Match quality: 25.86%
  Reason: 128 matching keypoints indicate these images
          likely show the same scene/object from different views.

MATCHING SUMMARY:
Found 1 MATCHING pair(s):

1. graf1 ↔ graf2
   → 128 keypoint matches
   → These images show similar content/scene
```

## Output Files

All results saved to `report/figures/gallery/`:

1. **`matching_report.txt`** 
   - Detailed explanation of which pairs match
   - Why each pair matches (or doesn't)
   
2. **`matches_matrix.csv`**
   - Pairwise match counts in table format
   - Easy to import into Excel/Google Sheets

3. **`matches_heatmap.jpg`**
   - Visual similarity matrix
   - Red = high matches, Blue = low matches

4. **Match visualizations** (one per pair)
   - `image1__image2_matches.jpg`
   - Shows matching keypoint pairs with lines

## Understanding the Results

### What Makes Images "Match"?

- **Threshold:** By default, pairs with **≥ 10 matching keypoints** are considered matching
- **Matching keypoints** = SIFT features that appear in both images
- **More matches** = more similar images

### Match Quality Examples:

- **100+ matches:** Excellent match (same scene, different viewpoint)
- **50-100 matches:** Good match (same object/scene with changes)
- **10-50 matches:** Weak match (some common features)
- **< 10 matches:** No match (different scenes)

### Why Do Images Match?

Images match when they have:
- Same scene from different angles
- Same object with rotation/scale changes
- Similar textures or patterns
- Overlapping content

Images DON'T match when they show:
- Completely different scenes
- Different objects
- No common features

## Advanced Options

```powershell
# Use different folder
python src/pairwise_match.py --folder path/to/images --out output/folder

# Change match threshold (default: 0.75)
python src/pairwise_match.py --folder data/gallery --out report/figures/gallery --ratio 0.8

# Change max matches in visualization (default: 200)
python src/pairwise_match.py --folder data/gallery --out report/figures/gallery --max_vis 100
```

## Example Workflow

```powershell
# 1. Place your 5 images in data/gallery/
# 2. Run matching
python src/pairwise_match.py --folder data/gallery --out report/figures/gallery

# 3. View results
start report/figures/gallery/matching_report.txt
start report/figures/gallery/matches_heatmap.jpg

# 4. View individual match visualizations
start report/figures/gallery/*.jpg
```

## Tips

- **Test with known pairs:** Start with 2 images you know are similar and 2 that are different
- **Image size:** Larger images take longer but may find more keypoints
- **Adjust threshold:** Edit line 71 in `src/pairwise_match.py` to change `match_threshold`

## Troubleshooting

**"No matching pairs found"**
- Lower the threshold (edit `match_threshold` in the script)
- Make sure you have similar images (same scene, different angle)
- Check that images are not too different

**"Too few keypoints detected"**
- Try images with more texture/details
- Avoid very dark or very bright images
- Increase image resolution

**"Script is slow"**
- Reduce `--max_vis` to 50
- Use smaller images (resize before running)
- Reduce number of images in folder

---

**Happy Matching! 🎯**
