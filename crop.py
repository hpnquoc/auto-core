import cv2

# Global variables
cropping = False  # True if cropping is being performed
start_point = (0, 0)  # Starting point of the cropping rectangle
end_point = (0, 0)  # Ending point of the cropping rectangle
image = None  # To hold the image being displayed

def click_and_crop(event, x, y, flags, param):
    global start_point, end_point, cropping

    # If the left mouse button was clicked, record the starting position
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        cropping = True

    # Check if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        cropping = False  # cropping is finished

        # Draw a rectangle around the region of interest
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Image", image)

def crop_image_with_mouse(img):
    global image
    image = img.copy()
    
    # Create a window and bind the function to window events
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_and_crop)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        
        # If the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = img.copy()

        # If the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # Crop the region of interest and return it
    if start_point != end_point:  # Ensure a region was selected
        cropped = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        return cropped
    return None

# Load three images
image1 = cv2.imread("/home/hpnquoc/auto-core/output/visnow/montreal_dataset/night/images/cam15/snow/resized/cam15_18-02_01-40.jpeg")
image2 = cv2.imread("/home/hpnquoc/auto-core/output/visnow/montreal_dataset/night/images/cam15/snow/output/cam15_18-02_01-40.jpeg")
image3 = cv2.imread("/home/hpnquoc/auto-core/output/visnow/montreal_dataset/night/images/cam15/snow/output_ada/cam15_18-02_01-40.jpeg")

# Check if images are loaded successfully
if image1 is None or image2 is None or image3 is None:
    print("Error: One or more images could not be loaded.")
    exit()

# Get the cropping rectangle from the first image
print("Select a region to crop from Image 1. Press 'c' to confirm or 'r' to reset.")
cropped_image1 = crop_image_with_mouse(image1)

# Crop the same region from the other images
if cropped_image1 is not None:
    x_start, y_start = start_point
    x_end, y_end = end_point
    
    cropped_image2 = image2[y_start:y_end, x_start:x_end]
    cropped_image3 = image3[y_start:y_end, x_start:x_end]
    
    # Display the cropped images
    cv2.imshow("Cropped Image 1", cropped_image1)
    cv2.imshow("Cropped Image 2", cropped_image2)
    cv2.imshow("Cropped Image 3", cropped_image3)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
