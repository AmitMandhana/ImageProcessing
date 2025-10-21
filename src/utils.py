"""utils.py
Helper functions: IO, plotting, image scaling, and visualization utilities.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from pathlib import Path


def read_gray(path: str) -> np.ndarray:
    """Read image as grayscale float32 [0, 1]"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


def read_color(path: str) -> np.ndarray:
    """Read image as BGR color"""
    return cv2.imread(path)


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize image by scale factor"""
    h, w = image.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle (degrees) around center"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness by multiplying pixel values"""
    adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return adjusted


def draw_keypoints_custom(img, keypoints, output_path=None):
    """Draw keypoints with size proportional to scale"""
    img_kp = cv2.drawKeypoints(img, keypoints, None, 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if output_path:
        cv2.imwrite(output_path, img_kp)
    return img_kp


def plot_matches_matplotlib(img1, kp1, img2, kp2, matches, save_path=None):
    """Plot matches using matplotlib for better control"""
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    
    # Create side-by-side image
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_rgb if len(img1_rgb.shape) == 3 else np.stack([img1_rgb]*3, axis=2)
    canvas[:h2, w1:] = img2_rgb if len(img2_rgb.shape) == 3 else np.stack([img2_rgb]*3, axis=2)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)
    
    # Draw matches
    for m in matches[:100]:  # Limit to 100 for clarity
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        pt2_shifted = (pt2[0] + w1, pt2[1])
        plt.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], 
                'g-', linewidth=0.5, alpha=0.5)
    
    plt.axis('off')
    plt.title(f'Matches: {len(matches)}')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_repeatability_curve(data, xlabel, ylabel, title, save_path=None):
    """Plot repeatability curve (e.g., matches vs rotation/scale)"""
    plt.figure(figsize=(8, 6))
    x_vals, y_vals = zip(*data)
    plt.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def save_results_table(results: List[dict], save_path: str):
    """Save evaluation results as formatted text table"""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SIFT Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"Image Pair: {r['pair']}\n")
            f.write(f"  Keypoints (img1): {r['kp1']}\n")
            f.write(f"  Keypoints (img2): {r['kp2']}\n")
            f.write(f"  Matches: {r['matches']}\n")
            f.write(f"  Match Ratio: {r['match_ratio']:.3f}\n")
            if 'runtime' in r:
                f.write(f"  Runtime: {r['runtime']:.3f}s\n")
            f.write("\n")


if __name__ == "__main__":
    print("utils module: image IO, transformations, and visualization utilities")
