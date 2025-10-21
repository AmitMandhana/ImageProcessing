"""
pairwise_match.py

Run SIFT-from-scratch on all images in a folder and perform pairwise matching.
Saves: match visualizations and a similarity (matches) matrix CSV + heatmap.

Usage:
  python src/pairwise_match.py --folder data/gallery --out report/figures/gallery
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import itertools
import csv
import os
import sys

# Ensure project root is on path when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sift_from_scratch import SIFT_Scratch, match_descriptors, draw_matches


def list_images(folder: Path):
    exts = {'.jpg', '.jpeg', '.png', '.ppm', '.pgm'}
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    return files


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def pairwise_match(folder: Path, out_dir: Path, ratio: float = 0.75, max_vis: int = 200):
    ensure_dir(out_dir)

    images = list_images(folder)
    if len(images) < 2:
        print(f"Need at least 2 images in {folder} (found {len(images)})")
        return

    print(f"Found {len(images)} images in {folder}")

    # Load images
    imgs = [cv2.imread(str(p)) for p in images]
    names = [p.stem for p in images]

    # Detect & compute once per image
    sift = SIFT_Scratch(num_octaves=3, scales_per_octave=3)
    keypoints = []
    descriptors = []

    for i, img in enumerate(imgs):
        if img is None:
            print(f"Warning: Could not read {images[i]}")
            keypoints.append([])
            descriptors.append(np.empty((0, 128), dtype=np.float32))
            continue
        kp, desc = sift.detect_and_compute(img)
        keypoints.append(kp)
        descriptors.append(desc)
        print(f"[{i}] {images[i].name}: keypoints={len(kp)}")

    # Pairwise matching
    n = len(images)
    match_counts = np.zeros((n, n), dtype=int)
    matching_pairs = []
    match_threshold = 10  # Minimum matches to consider a pair as matching
    
    print("\n" + "="*70)
    print("PAIRWISE MATCHING ANALYSIS")
    print("="*70 + "\n")

    for i, j in itertools.combinations(range(n), 2):
        desc_i = descriptors[i]
        desc_j = descriptors[j]
        matches = match_descriptors(desc_i, desc_j, ratio=ratio)
        num_matches = len(matches)
        match_counts[i, j] = num_matches
        match_counts[j, i] = num_matches
        
        # Determine if this is a matching pair
        is_match = num_matches >= match_threshold
        status = "✓ MATCHING" if is_match else "✗ NOT MATCHING"
        
        print(f"{names[i]} <-> {names[j]}")
        print(f"  Keypoints: {len(keypoints[i])} vs {len(keypoints[j])}")
        print(f"  Matches found: {num_matches}")
        print(f"  Status: {status}")
        
        if is_match:
            match_ratio = num_matches / min(len(keypoints[i]), len(keypoints[j])) if min(len(keypoints[i]), len(keypoints[j])) > 0 else 0
            print(f"  Match quality: {match_ratio:.2%}")
            print(f"  Reason: {num_matches} matching keypoints indicate these images")
            print(f"          likely show the same scene/object from different views.")
            matching_pairs.append((names[i], names[j], num_matches))
        print()

        # Save visualization
        vis = draw_matches(imgs[i], keypoints[i], imgs[j], keypoints[j], matches, max_matches=max_vis)
        out_path = out_dir / f"{names[i]}__{names[j]}_matches.jpg"
        cv2.imwrite(str(out_path), vis)

    # Print summary
    print("="*70)
    print("MATCHING SUMMARY")
    print("="*70 + "\n")
    
    if matching_pairs:
        print(f"Found {len(matching_pairs)} MATCHING pair(s):\n")
        for idx, (name1, name2, matches) in enumerate(matching_pairs, 1):
            print(f"{idx}. {name1} ↔ {name2}")
            print(f"   → {matches} keypoint matches")
            print(f"   → These images show similar content/scene\n")
    else:
        print(f"No matching pairs found (threshold: {match_threshold} matches)")
        print("Suggestion: Images may be too different, or try lowering the threshold.\n")
    
    # Diagonal as self matches (#desc)
    for i in range(n):
        match_counts[i, i] = len(descriptors[i])

    # Save CSV
    csv_path = out_dir / "matches_matrix.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([" "] + names)
        for i in range(n):
            writer.writerow([names[i]] + list(match_counts[i]))
    print(f"Saved matrix: {csv_path}")

    # Optional: heatmap via OpenCV (simple)
    norm = (match_counts.astype(np.float32) / (match_counts.max() + 1e-6) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_dir / "matches_heatmap.jpg"), heatmap)
    print(f"Saved heatmap: {out_dir / 'matches_heatmap.jpg'}")
    
    # Save detailed text report
    report_path = out_dir / "matching_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("SIFT PAIRWISE IMAGE MATCHING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total images analyzed: {n}\n")
        f.write(f"Total pairs tested: {len(list(itertools.combinations(range(n), 2)))}\n")
        f.write(f"Matching threshold: {match_threshold} keypoints\n\n")
        
        f.write("Images:\n")
        for i, name in enumerate(names, 1):
            f.write(f"  {i}. {name} ({len(keypoints[i-1])} keypoints)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("MATCHING PAIRS (>= {} matches)\n".format(match_threshold))
        f.write("="*70 + "\n\n")
        
        if matching_pairs:
            for idx, (name1, name2, matches) in enumerate(matching_pairs, 1):
                f.write(f"{idx}. {name1} ↔ {name2}\n")
                f.write(f"   Matches: {matches} keypoint pairs\n")
                f.write(f"   Explanation: These images share {matches} matching keypoints,\n")
                f.write(f"                indicating they likely show the same scene or object\n")
                f.write(f"                from different viewpoints or under different conditions.\n\n")
        else:
            f.write("No matching pairs found.\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("ALL PAIR DETAILS\n")
        f.write("="*70 + "\n\n")
        
        for i, j in itertools.combinations(range(n), 2):
            num_matches = match_counts[i, j]
            status = "MATCH" if num_matches >= match_threshold else "NO MATCH"
            f.write(f"{names[i]} <-> {names[j]}: {num_matches} matches [{status}]\n")
    
    print(f"Saved detailed report: {report_path}")
    print(f"\nAll results saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pairwise SIFT matching across a folder of images")
    parser.add_argument("--folder", type=str, default="data/gallery", help="Folder with images")
    parser.add_argument("--out", type=str, default="report/figures/gallery", help="Output directory")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    parser.add_argument("--max_vis", type=int, default=200, help="Max matches to visualize per pair")
    args = parser.parse_args()

    folder = Path(args.folder)
    out = Path(args.out)
    pairwise_match(folder, out, ratio=args.ratio, max_vis=args.max_vis)


if __name__ == "__main__":
    main()
