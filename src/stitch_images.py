# src/stitch_images.py
"""
Image stitching using SIFT keypoints and homography.
Combines matching images into a panorama.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sift_from_scratch import SIFT_Scratch, match_descriptors


def compute_homography_ransac(pts1, pts2, threshold=5.0, max_iters=2000):
    """
    Compute homography matrix using RANSAC.
    
    Args:
        pts1, pts2: matching points (N x 2 arrays)
        threshold: inlier threshold in pixels
        max_iters: maximum RANSAC iterations
    
    Returns:
        H: 3x3 homography matrix
        mask: inlier mask
    """
    best_H = None
    best_inliers = 0
    best_mask = None
    
    n = len(pts1)
    if n < 4:
        print("Error: Need at least 4 point matches for homography")
        return None, None
    
    for _ in range(max_iters):
        # Randomly sample 4 points
        idx = np.random.choice(n, 4, replace=False)
        src = pts1[idx]
        dst = pts2[idx]
        
        # Compute homography from 4 points
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        
        # Count inliers
        pts1_h = np.hstack([pts1, np.ones((n, 1))])  # homogeneous coordinates
        pts2_proj = (H @ pts1_h.T).T
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2:3]  # normalize
        
        errors = np.linalg.norm(pts2 - pts2_proj, axis=1)
        inliers = errors < threshold
        n_inliers = np.sum(inliers)
        
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_H = H
            best_mask = inliers
    
    # Refine H using all inliers
    if best_mask is not None and best_inliers >= 4:
        H_refined = cv2.getPerspectiveTransform(
            pts1[best_mask].astype(np.float32),
            pts2[best_mask].astype(np.float32)
        )
        return H_refined, best_mask
    
    return best_H, best_mask


def stitch_two_images(img1, img2, kp1, kp2, matches, use_opencv_homography=True):
    """
    Stitch two images together using their matches.
    
    Args:
        img1, img2: images to stitch
        kp1, kp2: keypoints from SIFT
        matches: DMatch objects
        use_opencv_homography: use OpenCV's findHomography (faster, more robust)
    
    Returns:
        panorama: stitched image
        H: homography matrix
        n_inliers: number of inlier matches
    """
    if len(matches) < 4:
        print(f"Error: Need at least 4 matches for stitching (got {len(matches)})")
        return None, None, 0
    
    # Extract matching point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Compute homography
    if use_opencv_homography:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    else:
        H, mask = compute_homography_ransac(pts1, pts2)
    
    if H is None:
        print("Error: Could not compute homography")
        return None, None, 0
    
    n_inliers = np.sum(mask) if mask is not None else 0
    
    print(f"  Homography computed: {n_inliers}/{len(matches)} inliers")
    
    # Warp img1 to img2's coordinate system
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Find corners of img1 after warping
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners1_warped = cv2.perspectiveTransform(corners1, H)
    
    # Find bounding box for the stitched image
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners1_warped, corners2], axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation to make all coordinates positive
    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])
    
    # Warp img1
    output_size = (x_max - x_min, y_max - y_min)
    img1_warped = cv2.warpPerspective(img1, translation @ H, output_size)
    
    # Place img2 in the output
    panorama = img1_warped.copy()
    y_offset = -y_min
    x_offset = -x_min
    
    # Blend img2 into panorama
    # Simple blending: overlay img2, keeping img1 where img2 is black
    roi = panorama[y_offset:y_offset+h2, x_offset:x_offset+w2]
    
    # Create mask where img2 has content
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    mask2 = (gray2 > 0).astype(np.uint8)
    
    # Blend using weighted average in overlap region
    if len(img2.shape) == 3:
        for c in range(3):
            roi_channel = roi[:, :, c]
            img2_channel = img2[:, :, c]
            
            # Where both images have content, average them
            overlap = (roi_channel > 0) & (img2_channel > 0)
            roi_channel[overlap] = (roi_channel[overlap].astype(np.float32) * 0.5 + 
                                   img2_channel[overlap].astype(np.float32) * 0.5).astype(np.uint8)
            
            # Where only img2 has content, use img2
            only_img2 = (roi_channel == 0) & (img2_channel > 0)
            roi_channel[only_img2] = img2_channel[only_img2]
            
            panorama[y_offset:y_offset+h2, x_offset:x_offset+w2, c] = roi_channel
    else:
        overlap = (roi > 0) & (img2 > 0)
        roi[overlap] = (roi[overlap].astype(np.float32) * 0.5 + 
                       img2[overlap].astype(np.float32) * 0.5).astype(np.uint8)
        only_img2 = (roi == 0) & (img2 > 0)
        roi[only_img2] = img2[only_img2]
        panorama[y_offset:y_offset+h2, x_offset:x_offset+w2] = roi
    
    return panorama, H, n_inliers


def stitch_image_pair(img1_path, img2_path, output_path, sift=None):
    """
    Complete pipeline to stitch two images.
    
    Args:
        img1_path, img2_path: paths to images
        output_path: where to save stitched result
        sift: SIFT detector (creates new if None)
    
    Returns:
        panorama: stitched image (or None if failed)
    """
    print(f"\n{'='*70}")
    print("IMAGE STITCHING")
    print(f"{'='*70}\n")
    
    # Load images
    print(f"Loading images...")
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return None
    
    print(f"  Image 1: {img1.shape}")
    print(f"  Image 2: {img2.shape}")
    
    # Detect keypoints and compute descriptors
    if sift is None:
        sift = SIFT_Scratch(num_octaves=3, scales_per_octave=3)
    
    print(f"\nDetecting SIFT keypoints...")
    kp1, desc1 = sift.detect_and_compute(img1)
    kp2, desc2 = sift.detect_and_compute(img2)
    
    print(f"  Image 1: {len(kp1)} keypoints")
    print(f"  Image 2: {len(kp2)} keypoints")
    
    # Match descriptors
    print(f"\nMatching descriptors...")
    matches = match_descriptors(desc1, desc2, ratio=0.75)
    print(f"  Found {len(matches)} matches")
    
    if len(matches) < 4:
        print("\nError: Not enough matches for stitching (need at least 4)")
        return None
    
    # Stitch images
    print(f"\nStitching images...")
    panorama, H, n_inliers = stitch_two_images(img1, img2, kp1, kp2, matches)
    
    if panorama is None:
        print("Error: Stitching failed")
        return None
    
    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panorama)
    
    print(f"\n{'='*70}")
    print("STITCHING SUCCESSFUL!")
    print(f"{'='*70}")
    print(f"  Panorama size: {panorama.shape}")
    print(f"  Inliers: {n_inliers}/{len(matches)}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*70}\n")
    
    return panorama


def auto_stitch_best_pair(folder_path, output_dir):
    """
    Automatically find the best matching pair and stitch them.
    
    Args:
        folder_path: folder containing images
        output_dir: where to save results
    """
    print(f"\n{'='*70}")
    print("AUTO-STITCHING: Finding best matching pair")
    print(f"{'='*70}\n")
    
    folder = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.pgm']
    images = []
    for ext in extensions:
        images.extend(folder.glob(f'*{ext}'))
        images.extend(folder.glob(f'*{ext.upper()}'))
    images = sorted(list(set(images)))
    
    if len(images) < 2:
        print("Error: Need at least 2 images in folder")
        return
    
    print(f"Found {len(images)} images")
    
    # Initialize SIFT
    sift = SIFT_Scratch(num_octaves=3, scales_per_octave=3)
    
    # Detect keypoints for all images
    print("\nDetecting keypoints...")
    all_kp = []
    all_desc = []
    all_img = []
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        kp, desc = sift.detect_and_compute(img)
        all_kp.append(kp)
        all_desc.append(desc)
        all_img.append(img)
        print(f"  {img_path.name}: {len(kp)} keypoints")
    
    # Find best matching pair
    print("\nFinding best matching pair...")
    best_matches = 0
    best_i, best_j = 0, 1
    
    from itertools import combinations
    for i, j in combinations(range(len(all_img)), 2):
        matches = match_descriptors(all_desc[i], all_desc[j], ratio=0.75)
        print(f"  {images[i].name} <-> {images[j].name}: {len(matches)} matches")
        if len(matches) > best_matches:
            best_matches = len(matches)
            best_i, best_j = i, j
    
    if best_matches < 4:
        print("\nError: No suitable pair found for stitching (need at least 4 matches)")
        return
    
    print(f"\nBest pair: {images[best_i].name} <-> {images[best_j].name}")
    print(f"  Matches: {best_matches}")
    
    # Stitch the best pair
    output_name = f"panorama_{images[best_i].stem}_{images[best_j].stem}.jpg"
    output_path = output_dir / output_name
    
    matches = match_descriptors(all_desc[best_i], all_desc[best_j], ratio=0.75)
    panorama, H, n_inliers = stitch_two_images(
        all_img[best_i], all_img[best_j],
        all_kp[best_i], all_kp[best_j],
        matches
    )
    
    if panorama is not None:
        cv2.imwrite(str(output_path), panorama)
        print(f"\nPanorama saved to: {output_path}")
    
    return panorama


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image stitching using SIFT")
    parser.add_argument("--img1", type=str, help="Path to first image")
    parser.add_argument("--img2", type=str, help="Path to second image")
    parser.add_argument("--folder", type=str, help="Auto-stitch best pair from folder")
    parser.add_argument("--output", type=str, default="report/figures/stitched",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.folder:
        auto_stitch_best_pair(args.folder, args.output)
    elif args.img1 and args.img2:
        output_path = Path(args.output) / "panorama.jpg"
        stitch_image_pair(args.img1, args.img2, output_path)
    else:
        print("Usage:")
        print("  Auto-stitch best pair: python src/stitch_images.py --folder data/gallery")
        print("  Stitch two images: python src/stitch_images.py --img1 img1.jpg --img2 img2.jpg")
