import cv2
import numpy as np

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

# Load two images (ensure they are the same size)
image1 = cv2.imread("/home/hpnquoc/auto-core/output/1.png")
image2 = cv2.imread("/home/hpnquoc/auto-core/output/3.png")

# Resize the images to the same dimensions if needed
if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

image1 = crop_image(image1, d=16)
image2 = crop_image(image2, d=16)

# Compute the absolute difference between the two images
difference = cv2.absdiff(image1, image2)

# Convert the difference to grayscale for better visualization
gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

# Apply a heatmap color map to the grayscale difference
heatmap = cv2.applyColorMap(gray_difference, cv2.COLORMAP_JET)

# Display the images and the difference using OpenCV
cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.imshow("Difference", heatmap)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()