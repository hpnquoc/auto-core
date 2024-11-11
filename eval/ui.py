import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir_input", type=str, default="/home/hpnquoc/auto-core/output/vehiclehah/resized")
parser.add_argument("--dir_output", type=str, default="/home/hpnquoc/auto-core/output/vehiclehah/output")
args = parser.parse_args()

class ImageComparer:
    def __init__(self, input_folder, output_folder):
        self.input_images = self.load_images_from_folder(input_folder)
        self.output_images = self.load_images_from_folder(output_folder)
        self.filenames = sorted(list(set(self.input_images.keys()).intersection(self.output_images.keys())))
        self.index = 0  # Start with the first image

    def load_images_from_folder(self, folder):
        """Load images from a folder and return a dictionary with filenames as keys and images as values."""
        images = {}
        for filename in os.listdir(folder):
            if filename.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                # img = np.array(img)  # Convert to NumPy array for OpenCV
                images[filename] = img
        return images

    def show_image(self):
        """Display the current pair of images side by side."""
        filename = self.filenames[self.index]
        input_img = self.input_images[filename]
        output_img = self.output_images[filename]

        # Resize images to the same height
        # input_img = cv2.resize(input_img, (400, 400))
        # output_img = cv2.resize(output_img, (400, 400))

        # Concatenate images side by side
        combined_img = np.hstack((input_img, output_img))

        # Display the image with OpenCV
        cv2.imshow(f"Comparison: {filename}", combined_img)

    def run(self):
        """Run the main loop for image comparison and navigation."""
        while True:
            # Close the OpenCV window
            cv2.destroyAllWindows()
            self.show_image()

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF

            # Navigate right (next image) with the right arrow key
            if key == ord('d') or key == 83:  # 'd' or right arrow
                self.index = (self.index + 1) % len(self.filenames)

            # Navigate left (previous image) with the left arrow key
            elif key == ord('a') or key == 81:  # 'a' or left arrow
                self.index = (self.index - 1) % len(self.filenames)

            # Exit the loop when 'q' or Esc is pressed
            elif key == ord('q') or key == 27:  # 'q' or Esc
                break
        

if __name__ == '__main__':
    # Specify the input and output folders
    input_folder = args.dir_input
    output_folder = args.dir_output

    # Create an ImageComparer instance and start the comparison
    comparer = ImageComparer(input_folder, output_folder)
    comparer.run()
