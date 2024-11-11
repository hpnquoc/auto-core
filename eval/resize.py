import os
import glob
import cv2
import numpy as np
import tqdm

def crop_image(img, d=32):
    """
    Crop the image to ensure dimensions are divisible by d.

    :param img: OpenCV image (numpy array)
    :param d: Integer, the divisor to make dimensions divisible by (default is 32)
    :return: Cropped OpenCV image (numpy array)
    """
    # Get the current dimensions of the image
    height, width = img.shape[:2]

    # Calculate new dimensions that are divisible by d
    new_width = width - width % d
    new_height = height - height % d

    # Calculate coordinates to crop the image evenly from the center
    x_start = (width - new_width) // 2
    y_start = (height - new_height) // 2
    x_end = x_start + new_width
    y_end = y_start + new_height

    # Crop the image
    img_cropped = img[y_start:y_end, x_start:x_end]
    return img_cropped

def get_all_images(folder_path):
    """
    Retrieve all image file paths from the given folder and its subfolders.

    :param folder_path: Path to the main folder
    :return: List of image file paths
    """
    # Define a list of common image file extensions
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff')

    # List to store image paths
    image_files = []

    # Iterate through each image extension and use glob to find matching files
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))

    return image_files

# Example usage
folder_path = 'output/'  # Specify the path to your folder
images = get_all_images(folder_path)

# Print the list of images found
for t in tqdm.tqdm(images):
    if '_ada' in t:
        img = cv2.imread(t)
        t = t.replace('_ada', '')
        t = t.replace('output', 'output_ada')
        os.makedirs(os.path.dirname(t), exist_ok=True)
        cv2.imwrite(t, img)
# for img_path in images:
#     img = cv2.imread(img_path)
#     img = crop_image(img, d=16)
#     cv2.imwrite(img_path, img)