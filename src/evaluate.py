"""evaluate.py
Comprehensive evaluation scripts for SIFT implementation.
Includes: repeatability tests, rotation/scale robustness, comparison with OpenCV SIFT.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sift_from_scratch import SIFT_Scratch, match_descriptors, draw_matches
from src.utils import (read_color, rotate_image, resize_image, adjust_brightness,
plot_repeatability_curve, save_results_table, draw_keypoints_custom)


def evaluate_single_pair(img1_path: str, img2_path: str, detector, output_dir: Path):
    """Evaluate SIFT on a single image pair"""
    img1 = read_color(img1_path)
    img2 = read_color(img2_path)
    
    # Detect and compute
    start = time.time()
    kp1, desc1 = detector.detect_and_compute(img1)
    kp2, desc2 = detector.detect_and_compute(img2)
    elapsed = time.time() - start
    
    # Match
    matches = match_descriptors(desc1, desc2, ratio=0.75)
    
    # Visualize
    img_kp1 = draw_keypoints_custom(img1, kp1, str(output_dir / "keypoints_img1.jpg"))
    img_kp2 = draw_keypoints_custom(img2, kp2, str(output_dir / "keypoints_img2.jpg"))
    img_matches = draw_matches(img1, kp1, img2, kp2, matches, max_matches=200)
    cv2.imwrite(str(output_dir / "matches.jpg"), img_matches)
    
    result = {
        'pair': f"{Path(img1_path).stem} vs {Path(img2_path).stem}",
        'kp1': len(kp1),
        'kp2': len(kp2),
        'matches': len(matches),
        'match_ratio': len(matches) / max(len(kp1), len(kp2), 1),
        'runtime': elapsed
    }
    
    print(f"Pair: {result['pair']}")
    print(f"  Keypoints: {result['kp1']}, {result['kp2']}")
    print(f"  Matches: {result['matches']}")
    print(f"  Runtime: {result['runtime']:.3f}s")
    
    return result


def evaluate_rotation_robustness(img_path: str, angles: List[float], detector, output_dir: Path):
    """Test rotation robustness by rotating image and measuring matches"""
    print("\n=== Rotation Robustness Test ===")
    img_orig = read_color(img_path)
    kp_orig, desc_orig = detector.detect_and_compute(img_orig)
    
    results = []
    for angle in angles:
        img_rot = rotate_image(img_orig, angle)
        kp_rot, desc_rot = detector.detect_and_compute(img_rot)
        matches = match_descriptors(desc_orig, desc_rot, ratio=0.75)
        results.append((angle, len(matches)))
        print(f"  Rotation {angle:3.0f}°: {len(matches)} matches")
    
    # Plot
    plot_repeatability_curve(results, "Rotation (degrees)", "Number of Matches",
                            "Rotation Robustness", str(output_dir / "rotation_curve.png"))
    return results


def evaluate_scale_robustness(img_path: str, scales: List[float], detector, output_dir: Path):
    """Test scale robustness by scaling image and measuring matches"""
    print("\n=== Scale Robustness Test ===")
    img_orig = read_color(img_path)
    kp_orig, desc_orig = detector.detect_and_compute(img_orig)
    
    results = []
    for scale in scales:
        img_scaled = resize_image(img_orig, scale)
        kp_scaled, desc_scaled = detector.detect_and_compute(img_scaled)
        matches = match_descriptors(desc_orig, desc_scaled, ratio=0.75)
        results.append((scale, len(matches)))
        print(f"  Scale {scale:.2f}x: {len(matches)} matches")
    
    # Plot
    plot_repeatability_curve(results, "Scale Factor", "Number of Matches",
                            "Scale Robustness", str(output_dir / "scale_curve.png"))
    return results


def evaluate_brightness_robustness(img_path: str, factors: List[float], detector, output_dir: Path):
    """Test brightness robustness"""
    print("\n=== Brightness Robustness Test ===")
    img_orig = read_color(img_path)
    kp_orig, desc_orig = detector.detect_and_compute(img_orig)
    
    results = []
    for factor in factors:
        img_bright = adjust_brightness(img_orig, factor)
        kp_bright, desc_bright = detector.detect_and_compute(img_bright)
        matches = match_descriptors(desc_orig, desc_bright, ratio=0.75)
        results.append((factor, len(matches)))
        print(f"  Brightness {factor:.2f}x: {len(matches)} matches")
    
    # Plot
    plot_repeatability_curve(results, "Brightness Factor", "Number of Matches",
                            "Brightness Robustness", str(output_dir / "brightness_curve.png"))
    return results


def compare_with_opencv(img1_path: str, img2_path: str, output_dir: Path):
    """Compare custom SIFT with OpenCV SIFT (if available)"""
    print("\n=== Comparison with OpenCV SIFT ===")
    img1 = read_color(img1_path)
    img2 = read_color(img2_path)
    
    # Custom SIFT
    detector_custom = SIFT_Scratch(num_octaves=3, scales_per_octave=3)
    start = time.time()
    kp1_custom, desc1_custom = detector_custom.detect_and_compute(img1)
    kp2_custom, desc2_custom = detector_custom.detect_and_compute(img2)
    time_custom = time.time() - start
    matches_custom = match_descriptors(desc1_custom, desc2_custom, ratio=0.75)
    
    # OpenCV SIFT (if available)
    try:
        detector_cv = cv2.SIFT_create()
        start = time.time()
        kp1_cv, desc1_cv = detector_cv.detectAndCompute(img1, None)
        kp2_cv, desc2_cv = detector_cv.detectAndCompute(img2, None)
        time_cv = time.time() - start
        matches_cv = match_descriptors(desc1_cv, desc2_cv, ratio=0.75)
        
        print(f"\nCustom SIFT:")
        print(f"  Keypoints: {len(kp1_custom)}, {len(kp2_custom)}")
        print(f"  Matches: {len(matches_custom)}")
        print(f"  Runtime: {time_custom:.3f}s")
        
        print(f"\nOpenCV SIFT:")
        print(f"  Keypoints: {len(kp1_cv)}, {len(kp2_cv)}")
        print(f"  Matches: {len(matches_cv)}")
        print(f"  Runtime: {time_cv:.3f}s")
        
        print(f"\nSpeedup: {time_custom / time_cv:.2f}x slower")
        
    except AttributeError:
        print("OpenCV SIFT not available (may require opencv-contrib-python)")


def run_full_evaluation(data_dir: Path, output_dir: Path):
    """Run complete evaluation pipeline"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector = SIFT_Scratch(num_octaves=3, scales_per_octave=3, 
                           contrast_thresh=0.04, edge_thresh=10)
    
    results = []
    
    # Evaluate image pairs
    pairs = [
        (data_dir / "graffiti" / "img1.jpg", data_dir / "graffiti" / "img2.jpg"),
        (data_dir / "bark" / "img1.jpg", data_dir / "bark" / "img2.jpg"),
    ]
    
    for img1, img2 in pairs:
        if img1.exists() and img2.exists():
            pair_dir = output_dir / f"{img1.parent.name}_pair"
            pair_dir.mkdir(exist_ok=True)
            result = evaluate_single_pair(str(img1), str(img2), detector, pair_dir)
            results.append(result)
    
    # Robustness tests (use first available image)
    test_img = None
    for p in pairs:
        if p[0].exists():
            test_img = p[0]
            break
    
    if test_img:
        robustness_dir = output_dir / "robustness"
        robustness_dir.mkdir(exist_ok=True)
        
        evaluate_rotation_robustness(str(test_img), [0, 15, 30, 45, 60, 90], 
                                     detector, robustness_dir)
        evaluate_scale_robustness(str(test_img), [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], 
                                 detector, robustness_dir)
        evaluate_brightness_robustness(str(test_img), [0.5, 0.75, 1.0, 1.25, 1.5], 
                                       detector, robustness_dir)
    
    # Comparison with OpenCV
    if pairs and pairs[0][0].exists() and pairs[0][1].exists():
        compare_with_opencv(str(pairs[0][0]), str(pairs[0][1]), output_dir)
    
    # Save results
    save_results_table(results, str(output_dir / "results_summary.txt"))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SIFT implementation")
    parser.add_argument("--data", type=str, default="data", help="Data directory")
    parser.add_argument("--output", type=str, default="report/figures", help="Output directory")
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    run_full_evaluation(data_dir, output_dir)
