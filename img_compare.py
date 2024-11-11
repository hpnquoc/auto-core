import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QGroupBox, QGridLayout

from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QKeyEvent
from PyQt6.QtCore import Qt
import os
import glob
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--dir_img1", type=str, default="")
argparser.add_argument("--dir_img2", type=str, default="")
# /home/hpnquoc/auto-core/output/visnow/montreal_dataset/day/images/cam1/snow/resized
# /home/hpnquoc/auto-core/output/visnow/montreal_dataset/day/images/cam1/snow/output

class ImageComparisonApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Comparison App")
        self.setGeometry(100, 100, 800, 400)
        self.setAcceptDrops(True)  # Enable drag-and-drop on the main window

        # Initialize paths for images and directories
        self.image1_name = None
        self.image2_name = None
        self.image1_folder = None
        self.image2_folder = None
        self.img_list = []

        # Central widget setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Buttons for loading and comparing images
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Images")
        self.load_button.clicked.connect(self.load_images)
        button_layout.addWidget(self.load_button)

        self.compare_button = QPushButton("Compare Images")
        self.compare_button.clicked.connect(self.compare_images)
        self.compare_button.setEnabled(False)
        button_layout.addWidget(self.compare_button)

        self.clear_button = QPushButton("Clear Images")
        self.clear_button.clicked.connect(self.clear_images)
        button_layout.addWidget(self.clear_button)

        self.main_layout.addLayout(button_layout)

        # Layout to display images
        self.setup_image_display()
        self.setup_key_navigation()

    def setup_image_display(self):
        # Group for image display
        image_group = QGroupBox("Image Comparison")
        image_layout = QGridLayout()

        # Image 1 display
        self.image1_label = QLabel("Image 1")
        self.image1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image1_label.setStyleSheet("border: 1px solid black; padding: 10px;")
        image_layout.addWidget(self.image1_label, 0, 0)

        # Image 2 display
        self.image2_label = QLabel("Image 2")
        self.image2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image2_label.setStyleSheet("border: 1px solid black; padding: 10px;")
        image_layout.addWidget(self.image2_label, 0, 1)

        # Difference result display
        self.result_label = QLabel("Difference")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("border: 1px solid black; padding: 10px;")
        image_layout.addWidget(self.result_label, 0, 2)

        image_group.setLayout(image_layout)
        self.main_layout.addWidget(image_group)
    def setup_key_navigation(self):
        # Instructions for keyboard navigation
        navigation_group = QGroupBox("Navigation")
        navigation_layout = QVBoxLayout()

        instructions = QLabel("Use A and D keys to navigate between images.\nPress 'Q' to quit.")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("font-size: 12px; color: gray; padding: 10px;")
        navigation_layout.addWidget(instructions)

        navigation_group.setLayout(navigation_layout)
        self.main_layout.addWidget(navigation_group)

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Accept the drag event if the dragged data is a file with image format
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        # Handle the drop event to set the image paths
        urls = event.mimeData().urls()
        if len(urls) > 0:
            # Process the first URL in case of multiple files being dragged
            file_path = urls[0].toLocalFile()
            if not self.image1_name:
                self.image1_name = os.path.basename(file_path)
                self.image1_folder = os.path.dirname(file_path)
                self.loadFileList()
                self.display_images()
            elif not self.image2_name:
                self.image2_name = os.path.basename(file_path)
                self.image2_folder = os.path.dirname(file_path)
                self.display_images()
                self.compare_button.setEnabled(True)
                self.compare_images()

    def load_images(self):
        # Open file dialog to select two images
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if len(file_paths) == 2:
                self.image1_name, self.image2_name = file_paths
                self.image1_folder = os.path.dirname(self.image1_name)
                self.image2_folder = os.path.dirname(self.image2_name)
                self.loadFileList()
                self.display_images()
                self.compare_button.setEnabled(True)
                self.compare_images()
            elif len(file_paths) == 1:
                if self.image1_name is None:
                    self.image1_name = file_paths[0]
                    self.image1_folder = os.path.dirname(self.image1_name)
                    self.loadFileList()
                    self.display_images()
                else:
                    self.image2_name = file_paths[0]
                    self.image2_folder = os.path.dirname(self.image2_name)
                    self.display_images()
                    self.compare_button.setEnabled(True)
                    self.compare_images()

    def loadFileList(self): 
        self.img_list = sorted(glob.glob(self.image1_folder + '/*.*'))
        self.img_list = [os.path.basename(img) for img in self.img_list]      

    def display_images(self):
        # Load and display image 1
        if self.image1_name:
            pixmap1 = QPixmap(os.path.join(self.image1_folder, self.image1_name)).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image1_label.setPixmap(pixmap1)

        # Load and display image 2
        if self.image2_name:
            pixmap2 = QPixmap(os.path.join(self.image2_folder, self.image2_name)).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image2_label.setPixmap(pixmap2)

        # Clear any previous result
        self.result_label.clear()

    def compare_images(self):
        # Load images as grayscale for comparison
        img1 = cv2.imread(os.path.join(self.image1_folder, self.image1_name), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(self.image2_folder, self.image2_name), cv2.IMREAD_GRAYSCALE)

        # Resize images to the same shape if they differ
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Calculate absolute difference
        diff = cv2.absdiff(img1, img2)

        # Apply a threshold to highlight differences
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Convert to QImage to display in PyQt6
        height, width = thresh.shape
        bytes_per_line = width
        q_image = QImage(thresh.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

        # Set result image to QLabel
        pixmap = QPixmap.fromImage(q_image).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.result_label.setPixmap(pixmap)

    def clear_images(self):
        # Clear image paths and labels
        self.image1_name = None
        self.image2_name = None
        self.image1_label.clear()
        self.image2_label.clear()
        self.result_label.clear()
        self.compare_button.setEnabled(False)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        key_map = {
            Qt.Key.Key_Right: 'D', Qt.Key.Key_Left: 'A', Qt.Key.Key_Q: 'Q', Qt.Key.Key_A: 'A', Qt.Key.Key_D: 'D'
        }
        if key in key_map:
            if key_map[key] == 'Q':
                self.close()
            elif key_map[key] == 'A':
                idx = (self.img_list.index(self.image1_name) - 1) % len(self.img_list)

                self.image1_name = self.img_list[idx]
                self.image2_name = self.img_list[idx]
                self.display_images()
                self.compare_images()
            elif key_map[key] == 'D':
                idx = (self.img_list.index(self.image1_name) + 1) % len(self.img_list)

                self.image1_name = self.img_list[idx]
                self.image2_name = self.img_list[idx]
                self.display_images()
                self.compare_images()

if __name__ == "__main__":
    arg = argparser.parse_args()
    app = QApplication(sys.argv)
    window = ImageComparisonApp()
    if arg.dir_img1 and arg.dir_img2:
        window.image1_folder = arg.dir_img1
        window.image2_folder = arg.dir_img2
        window.loadFileList()
        window.image1_name = window.img_list[0]
        window.image2_name = window.img_list[0]
        window.display_images()
        window.compare_images()
    window.show()
    sys.exit(app.exec())
