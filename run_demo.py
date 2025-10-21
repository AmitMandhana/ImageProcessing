# run_demo.py
"""
Quick demo script to run SIFT on image pairs and save visualizations.
Usage: python run_demo.py
"""

import cv2
from pathlib import Path
from src.sift_from_scratch import SIFT_Scratch, match_descriptors, draw_matches

def run_pair(img_path1, img_path2, out_path="matches_out.jpg"):
    """Run SIFT on a single image pair"""
    print(f"\nProcessing pair: {Path(img_path1).name} vs {Path(img_path2).name}")
    
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not read images")
        return
    
    # Create SIFT detector
    sift = SIFT_Scratch(num_octaves=3, scales_per_octave=3, 
                       contrast_thresh=0.04, edge_thresh=10)
    
    # Detect and compute
    print("Detecting keypoints and computing descriptors...")
    kp1, desc1 = sift.detect_and_compute(img1)
    kp2, desc2 = sift.detect_and_compute(img2)
    
    print(f"Keypoints detected: {len(kp1)} (img1), {len(kp2)} (img2)")
    
    # Match descriptors
    print("Matching descriptors...")
    matches = match_descriptors(desc1, desc2, ratio=0.75)
    print(f"Matches found: {len(matches)}")
    
    # Visualize and save
    vis = draw_matches(img1, kp1, img2, kp2, matches, max_matches=200)
    cv2.imwrite(out_path, vis)
    print(f"Saved visualization to: {out_path}")
    
    return len(kp1), len(kp2), len(matches)

def find_image_pair(directory):
    """Find any two image files in a directory (supports .jpg, .png, .ppm)"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.pgm']
    images = []
    for ext in image_extensions:
        images.extend(list(directory.glob(f'*{ext}')))
        images.extend(list(directory.glob(f'*{ext.upper()}')))
    
    # Remove duplicates and sort
    images = sorted(list(set(images)))[:2]  # Take first two unique images
    return images if len(images) >= 2 else None

def main():
    """Main demo function"""
    base = Path(__file__).parent
    output_dir = base / "report" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to find image pairs in data directories
    data_dirs = [
        (base / "data" / "graffiti", "graffiti_matches.jpg"),
        (base / "data" / "bark", "bark_matches.jpg"),
    ]
    
    print("="*60)
    print("SIFT from Scratch - Demo")
    print("="*60)
    
    found_pair = False
    for img_dir, out_name in data_dirs:
        if img_dir.exists():
            pair = find_image_pair(img_dir)
            if pair and len(pair) >= 2:
                found_pair = True
                print(f"\nFound images in {img_dir.name}/:")
                print(f"  - {pair[0].name}")
                print(f"  - {pair[1].name}")
                out_path = output_dir / out_name
                run_pair(str(pair[0]), str(pair[1]), str(out_path))
            else:
                print(f"\nSkipping {img_dir.name}/ (less than 2 images found)")
        else:
            print(f"\nSkipping {img_dir.name}/ (directory not found)")
    
    if not found_pair:
        print("\n" + "="*60)
        print("No image pairs found!")
        print("Please download images and place them in:")
        print("  - data/graffiti/img1.jpg and img2.jpg")
        print("  - data/bark/img1.jpg and img2.jpg")
        print("\nDataset: https://www.robots.ox.ac.uk/~vgg/research/affine/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print(f"Check {output_dir} for results")
        print("="*60)

if __name__ == "__main__":
    main()
