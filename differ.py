import os
import cv2
import argparse

def load_images_from_folder(folder):
    """Load and sort images from a folder."""
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

def compare_and_save_differences(dir_img1, dir_img2, result_dir):
    """Compare images from two directories and save the differences."""
    # Load and sort images
    img_list1 = load_images_from_folder(dir_img1)
    img_list2 = load_images_from_folder(dir_img2)

    # Ensure result directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Compare images pairwise
    for img1_path, img2_path in zip(img_list1, img_list2):
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)

        # Read images in grayscale
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Skipping {img1_name} or {img2_name} due to loading error.")
            continue

        # Resize images to match sizes if necessary
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute absolute difference
        diff = cv2.absdiff(img1, img2)

        # Apply threshold to highlight differences
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        # Save the result
        diff_name = f"diff_{img1_name}"
        diff_path = os.path.join(result_dir, diff_name)
        cv2.imwrite(diff_path, thresh)

        print(f"Saved difference image: {diff_path}")

if __name__ == "__main__":
    # Argument parser for directories
    parser = argparse.ArgumentParser(description="Compare images in two directories and save differences.")
    parser.add_argument("--dir_img1", type=str, required=True, help="Path to the first image directory")
    parser.add_argument("--dir_img2", type=str, required=True, help="Path to the second image directory")
    parser.add_argument("--result_dir", type=str, default="result", help="Path to save the result images")
    args = parser.parse_args()

    # Run the comparison
    compare_and_save_differences(args.dir_img1, args.dir_img2, args.result_dir)
