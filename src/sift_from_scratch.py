# src/sift_from_scratch.py
"""
Complete SIFT implementation from scratch.
Implements: Gaussian pyramid, DoG, scale-space extrema detection, keypoint refinement,
orientation assignment (36-bin histogram), 128-d descriptor (4×4 × 8 bins),
normalization and ratio-test matching.
"""

import cv2
import numpy as np
from typing import List, Tuple
import math

# ---------- Utilities ----------
def gray_and_float(img):
    """Convert image to grayscale float32 [0, 1]"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    g = g.astype(np.float32) / 255.0
    return g

def gaussian_kernel(sigma, ksize=None):
    """Generate a Gaussian kernel"""
    if ksize is None:
        ksize = int(2 * math.ceil(3 * sigma) + 1)
    ax = np.arange(-ksize//2 + 1., ksize//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def convolve(img, kernel):
    """Convolve image with kernel"""
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

# ---------- SIFT-like class ----------
class SIFT_Scratch:
    """
    Complete SIFT implementation from scratch.
    
    Runtime Complexity:
    - Gaussian pyramid: O(N * num_octaves * scales) where N = image pixels
    - DoG pyramid: O(N * num_octaves * scales)
    - Extrema detection: O(N * num_octaves * scales)
    - Descriptor computation: O(K * 256) where K = number of keypoints
    - Matching: O(K1 * K2) for brute-force
    """
    
    def __init__(self, num_octaves=4, scales_per_octave=3, sigma=1.6, 
                 contrast_thresh=0.04, edge_thresh=10):
        self.num_octaves = num_octaves
        self.scales = scales_per_octave
        self.sigma = sigma
        self.k = 2 ** (1.0 / scales_per_octave)
        self.contrast_thresh = contrast_thresh
        self.edge_thresh = edge_thresh

    def build_gaussian_pyramid(self, img: np.ndarray) -> List[List[np.ndarray]]:
        """
        Build Gaussian pyramid with num_octaves octaves and scales+3 scales per octave.
        Uses OpenCV GaussianBlur for efficiency.
        """
        base = img.copy()
        pyramid = []
        for o in range(self.num_octaves):
            octave_images = []
            for s in range(self.scales + 3):  # +3 as in Lowe's SIFT
                sigma_eff = self.sigma * (self.k ** s) * (2 ** o)
                # Use OpenCV GaussianBlur for speed with computed sigma
                blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma_eff, sigmaY=sigma_eff, 
                                          borderType=cv2.BORDER_REPLICATE)
                octave_images.append(blurred)
            pyramid.append(octave_images)
            # downsample base for next octave
            base = cv2.resize(base, (base.shape[1] // 2, base.shape[0] // 2), 
                            interpolation=cv2.INTER_NEAREST)
        return pyramid

    def build_dog_pyramid(self, g_pyr: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """Build Difference-of-Gaussian pyramid from Gaussian pyramid"""
        dog_pyr = []
        for octave in g_pyr:
            dog_oct = []
            for i in range(1, len(octave)):
                dog = octave[i] - octave[i - 1]
                dog_oct.append(dog)
            dog_pyr.append(dog_oct)
        return dog_pyr

    def detect_scale_space_extrema(self, dog_pyr: List[List[np.ndarray]]) -> List[Tuple[int,int,int,int]]:
        """
        Detect scale-space extrema (maxima and minima) in DoG pyramid.
        Returns list of (octave, scale, y, x) tuples.
        """
        keypoints = []  # list of (octave, scale, y, x)
        for o_idx, dog_oct in enumerate(dog_pyr):
            # we can only check interior scales (1 .. S) because dog_oct length = scales+2
            for s in range(1, len(dog_oct) - 1):
                prev_img = dog_oct[s - 1]
                cur_img = dog_oct[s]
                next_img = dog_oct[s + 1]
                # For speed we iterate pixels while skipping image boundary
                h, w = cur_img.shape
                # simple threshold to skip low-contrast pixels
                candidates = np.where(np.abs(cur_img) > self.contrast_thresh)
                for y, x in zip(candidates[0], candidates[1]):
                    if y < 1 or y >= h - 1 or x < 1 or x >= w - 1:
                        continue
                    val = cur_img[y, x]
                    patch_prev = prev_img[y-1:y+2, x-1:x+2]
                    patch_cur = cur_img[y-1:y+2, x-1:x+2]
                    patch_next = next_img[y-1:y+2, x-1:x+2]
                    stacked = np.stack([patch_prev, patch_cur, patch_next])
                    # check for maximum or minimum
                    if val == stacked.max() or val == stacked.min():
                        keypoints.append((o_idx, s, y, x))
        return keypoints

    def refine_keypoints(self, keypoints, dog_pyr):
        """
        Refine keypoints by removing edge-like points using Hessian principal curvature test.
        This removes points with high edge response (low corner response).
        """
        refined = []
        for (o, s, y, x) in keypoints:
            D = dog_pyr[o][s]
            if y <= 0 or x <= 0 or y >= D.shape[0]-1 or x >= D.shape[1]-1:
                continue
            # compute Hessian approx (2D) on current scale
            Dxx = D[y, x+1] + D[y, x-1] - 2 * D[y, x]
            Dyy = D[y+1, x] + D[y-1, x] - 2 * D[y, x]
            Dxy = ((D[y+1, x+1] - D[y+1, x-1]) - (D[y-1, x+1] - D[y-1, x-1])) / 4.0
            tr = Dxx + Dyy
            det = Dxx * Dyy - Dxy * Dxy
            if det <= 0:
                continue
            r = (tr * tr) / det
            if r < ((self.edge_thresh + 1)**2) / self.edge_thresh:
                refined.append((o, s, y, x))
        return refined

    def assign_orientations(self, g_pyr, keypoints):
        """
        Assign orientation to keypoints using 36-bin histogram of gradient orientations.
        Returns keypoints with (octave, scale, y, x, angle).
        Can return multiple orientations for the same keypoint location.
        """
        oriented = []
        for (o, s, y, x) in keypoints:
            img = g_pyr[o][s]
            h, w = img.shape
            if y <= 0 or x <= 0 or y >= h-1 or x >= w-1:
                continue
            # 16x16 window used to build histogram
            radius = int(round(3 * 1.5))  # 1.5 ~ scale-dependent (simplified)
            hist = np.zeros(36, dtype=np.float32)
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    yy, xx = y + dy, x + dx
                    if yy <= 0 or xx <= 0 or yy >= h-1 or xx >= w-1:
                        continue
                    mag, ang = self._grad_mag_ang(img, yy, xx)
                    bin_idx = int(np.round((ang % (2*np.pi)) * 36 / (2*np.pi))) % 36
                    hist[bin_idx] += mag
            # find dominant peaks (within 80% of max we also keep extra orientations)
            maxv = hist.max()
            if maxv <= 0:
                continue
            peaks = np.where(hist >= 0.8 * maxv)[0]
            for p in peaks:
                angle = (p + 0.5) * (2*np.pi / 36)
                oriented.append((o, s, y, x, angle))
        return oriented

    def _grad_mag_ang(self, img, y, x):
        """Compute gradient magnitude and angle at pixel (y, x)"""
        dx = img[y, x+1] - img[y, x-1]
        dy = img[y-1, x] - img[y+1, x]
        mag = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) % (2*np.pi)
        return mag, ang

    def compute_descriptors(self, g_pyr, oriented_keypoints):
        """
        Compute 128-d SIFT descriptors for oriented keypoints.
        Descriptor: 4×4 spatial grid × 8 orientation bins = 128 dimensions.
        Includes normalization and clipping for illumination invariance.
        """
        descriptors = []
        kps_out = []
        for (o, s, y, x, angle) in oriented_keypoints:
            img = g_pyr[o][s]
            h, w = img.shape
            # descriptor window: 16x16 patch, rotated by -angle
            half_w = 8
            bins = 8
            desc = np.zeros((4, 4, bins), dtype=np.float32)
            cos_t = math.cos(-angle)
            sin_t = math.sin(-angle)
            for i in range(-half_w, half_w):
                for j in range(-half_w, half_w):
                    # rotate point (i,j) -> sample coordinates
                    rx = int(round(x + cos_t * j - sin_t * i))
                    ry = int(round(y + sin_t * j + cos_t * i))
                    if ry <= 0 or rx <= 0 or ry >= h-1 or rx >= w-1:
                        continue
                    mag, ang = self._grad_mag_ang(img, ry, rx)
                    # relative coordinates to center mapped into 4x4 cells
                    bin_y = int(np.floor((i + half_w) / 4.0))
                    bin_x = int(np.floor((j + half_w) / 4.0))
                    if bin_x < 0 or bin_x >= 4 or bin_y < 0 or bin_y >= 4:
                        continue
                    # orientation relative to keypoint angle
                    rel_ang = (ang - angle) % (2*np.pi)
                    bin_ori = int(np.floor(rel_ang / (2*np.pi / bins))) % bins
                    desc[bin_y, bin_x, bin_ori] += mag
            vec = desc.flatten()
            # normalize, threshold, renormalize (for illumination invariance)
            if np.linalg.norm(vec) < 1e-6:
                continue
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            vec = np.clip(vec, 0, 0.2)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            descriptors.append(vec.astype(np.float32))
            kps_out.append((o, s, y, x, angle))
        return np.array(kps_out), np.array(descriptors)

    def detect_and_compute(self, img_bgr: np.ndarray):
        """
        Main pipeline: detect keypoints and compute descriptors.
        Returns OpenCV-compatible KeyPoint objects and descriptor array.
        """
        img = gray_and_float(img_bgr)
        g_pyr = self.build_gaussian_pyramid(img)
        dog_pyr = self.build_dog_pyramid(g_pyr)
        raw_kps = self.detect_scale_space_extrema(dog_pyr)
        refined = self.refine_keypoints(raw_kps, dog_pyr)
        oriented = self.assign_orientations(g_pyr, refined)
        kps, desc = self.compute_descriptors(g_pyr, oriented)
        # convert keypoints list to OpenCV-like KeyPoint objects for visualization convenience
        out_kps = []
        for (o, s, y, x, a) in kps:
            scale = self.sigma * (self.k ** s) * (2 ** o)
            kp = cv2.KeyPoint(
                x=float(x * (2 ** o)), 
                y=float(y * (2 ** o)), 
                size=float(scale*6),  # Fixed: use 'size' not '_size'
                angle=float(a*180/math.pi)  # Fixed: use 'angle' not '_angle'
            )
            out_kps.append(kp)
        return out_kps, desc

# ---------- Matching & visualization ----------
def match_descriptors(desc1, desc2, ratio=0.75):
    """
    Match descriptors using brute-force with L2 distance and Lowe's ratio test.
    Ratio test: accept match if distance to nearest neighbor is < ratio * distance to second nearest.
    Runtime: O(N1 * N2) for brute-force.
    """
    if desc1 is None or desc2 is None or len(desc1)==0 or len(desc2)==0:
        return []
    dists = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)  # (N1, N2)
    idx1 = np.arange(dists.shape[0])
    matches = []
    for i in idx1:
        sorted_idx = np.argsort(dists[i])
        if len(sorted_idx) < 2: continue
        if dists[i, sorted_idx[0]] < ratio * dists[i, sorted_idx[1]]:
            m = cv2.DMatch(
                _queryIdx=int(i), 
                _trainIdx=int(sorted_idx[0]), 
                _distance=float(dists[i, sorted_idx[0]])
            )
            matches.append(m)
    return matches

def draw_matches(img1, kp1, img2, kp2, matches, max_matches=200):
    """Draw matches between two images"""
    out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return out

# ---------- Example quick test ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sift_from_scratch.py <img1> <img2>")
        sys.exit(1)
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    detector = SIFT_Scratch()
    kp1, desc1 = detector.detect_and_compute(img1)
    kp2, desc2 = detector.detect_and_compute(img2)
    matches = match_descriptors(desc1, desc2, ratio=0.75)
    out = draw_matches(img1, kp1, img2, kp2, matches)
    cv2.imwrite("matches_demo.jpg", out)
    print(f"kp1={len(kp1)}, kp2={len(kp2)}, matches={len(matches)}")
