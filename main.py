import math

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from PyQt5.QtWidgets import QLabel, QRadioButton
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QButtonGroup, QPushButton, QSlider, QFileDialog
from PyQt5.QtWidgets import QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage
import time


from PyQt5.QtWidgets import QButtonGroup, QRadioButton


def generate_heatmap_pixmap(R, threshold_ratio=0.01):
    # Create figure offscreen
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(R, cmap='hot')
    ax.set_title("Corner Response $R$")
    ax.axis('off')

    # Add colorbar
    cbar = plt.colorbar(ax.imshow(R, cmap='hot'), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Corner Response')

    # Highlight corners
    threshold = R.max() * threshold_ratio
    corners_y, corners_x = np.where(R > threshold)
    ax.plot(corners_x, corners_y, 'o', markerfacecolor='none', markeredgecolor='yellow', markersize=1)

    # Save figure to memory
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    # Convert to QPixmap
    pil_image = Image.open(buf)
    qt_image = pil_image.convert("RGBA")
    data = qt_image.tobytes("raw", "RGBA")
    qimage = QPixmap()
    qimage.loadFromData(buf.getvalue())
    return qimage


def check_if_grayscale(img):
    if img is None or not isinstance(img, np.ndarray):
        return False
    if len(img.shape) == 2:
        return True
    if len(img.shape) == 3 and img.shape[2] == 3:
        return np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2])
    return False


# def cv_to_qpixmap(cv_image):  # replaced this with plotting
#     if len(cv_image.shape) == 2:  # Grayscale image
#         self.height, self.width = cv_image.shape
#         bytes_per_line = self.width
#         qimage = QImage(cv_image.data, self.width, self.height, bytes_per_line, QImage.Format_Grayscale8)
#     elif len(cv_image.shape) == 3:  # Color image
#         self.height, self.width, channels = cv_image.shape
#         if channels == 3:
#             # Convert BGR to RGB
#             cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
#             bytes_per_line = channels * self.width
#             qimage = QImage(cv_image_rgb.data, self.width, self.height, bytes_per_line, QImage.Format_RGB888)
#         else:
#             raise ValueError(f"Unsupported number of channels: {channels}")
#     else:
#         raise ValueError("Unsupported image format")
#     return QPixmap.fromImage(qimage)


def apply_smoothing_window(component, height, width, kernel_size=3):
    if kernel_size == 1:
        kernel_size = 3
    elif kernel_size == 2:
        kernel_size = 5
    elif kernel_size == 3:
        kernel_size = 7
    print(f"Kernel Size: {kernel_size}")
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    smoothed = np.copy(component)
    for i in range(kernel_size // 2, height - kernel_size // 2):
        for j in range(kernel_size // 2, width - kernel_size // 2):
            neighbors = component[i - kernel_size // 2:i + kernel_size // 2 + 1,
                                  j - kernel_size // 2:j + kernel_size // 2 + 1]
            smoothed[i, j] = np.sum(neighbors * kernel)
    return smoothed


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("MainWindow.ui", self)
        self.setWindowTitle("SIFTIFY")
        self.setWindowIcon(QIcon("icon.png"))

        self.image_path = None
        self.original_cv_image = None
        self.grayscale_org_image = None
        self.height = None
        self.width = None
        self.original_image_label = self.findChild(QLabel, "Widget_Output_3")
        self.original_image_label.setScaledContents(True)
        self.output_image_label = self.findChild(QLabel, "Widget_Output_1")
        self.corner_radio_btn = self.findChild(QRadioButton, "RadioButton_Harris_2")
        self.corner_radio_btn.toggled.connect(self.activate_corner_detection)

        self.harris_radio_btn = self.findChild(QRadioButton, "radioButton_3")
        self.harris_radio_btn.toggled.connect(self.apply_harris_operator)
        self.harris_radio_btn.setEnabled(False)

        self.lambda_radio_btn = self.findChild(QRadioButton, "radioButton_4")
        self.lambda_radio_btn.toggled.connect(self.apply_lambda_minus)
        self.lambda_radio_btn.setEnabled(False)

        self.sift_button = self.findChild(QRadioButton, "RadioButton_SIFT_2")
        self.match_button = self.findChild(QRadioButton, "RadioButton_Match_3")
        self.ssd_button = self.findChild(QRadioButton, "RadioButton_SSD_2")
        self.nnd_button = self.findChild(QRadioButton, "RadioButton_NND_2")


        corner_detection_group = QButtonGroup(self)
        corner_detection_group.setExclusive(True)
        corner_detection_group.addButton(self.harris_radio_btn)
        corner_detection_group.addButton(self.lambda_radio_btn)

        match_group = QButtonGroup(self)
        match_group.setExclusive(True)
        match_group.addButton(self.ssd_button)
        match_group.addButton(self.nnd_button)

        main_group = QButtonGroup(self)
        main_group.setExclusive(True)
        main_group.addButton(self.corner_radio_btn)
        main_group.addButton(self.sift_button)
        main_group.addButton(self.match_button)

        self.k_label = self.findChild(QLabel, "label_param_6")
        self.k_parameter = self.findChild(QSlider, "slider_Alpha_4")
        self.k_parameter.setRange(1, 20)
        self.k_parameter.setValue(4)
        self.k_parameter.valueChanged.connect(self.k_value_changed)

        self.harris_thres_label = self.findChild(QLabel, "label_param_8")
        self.harris_threshold = self.findChild(QSlider, "slider_Alpha_6")
        self.harris_threshold.setRange(1, 10)
        self.harris_threshold.setValue(1)
        self.harris_threshold.valueChanged.connect(self.harris_thres_slider_value_changed)

        self.harris_kernel_label = self.findChild(QLabel, "label_param_10")
        self.harris_kernel = self.findChild(QSlider, "slider_Alpha_8")
        self.harris_kernel.setRange(1, 3)
        self.harris_kernel.setValue(1)
        self.harris_kernel.valueChanged.connect(self.harris_kernel_value_changed)

        self.lambda_thres_label = self.findChild(QLabel, "label_param_11")
        self.lambda_threshold = self.findChild(QSlider, "slider_Alpha_9")
        self.lambda_threshold.setRange(1, 10)
        self.lambda_threshold.setValue(1)
        self.lambda_threshold.valueChanged.connect(self.lambda_thres_slider_value_changed)

        self.lambda_kernel_label = self.findChild(QLabel, "label_param_12")
        self.lambda_kernel = self.findChild(QSlider, "slider_Alpha_10")
        self.lambda_kernel.setRange(1, 3)
        self.lambda_kernel.setValue(1)
        self.lambda_kernel.valueChanged.connect(self.lambda_kernel_value_changed)

        self.done_button = self.findChild(QPushButton, "pushButton_done_2")
        self.done_button.clicked.connect(self.apply_corner_detection)



#///////////////////////////////////////////////////////////////////////////////////////////

        self.image=None
        self.start_time_cal_sift=0
        self.elapsed_time_cal_sift=0
        self.upload_first=False
        self.pushButton_Upload_Img1.clicked.connect(lambda : self.Upload_Imgs(1))
        self.pushButton_Upload_Img2.clicked.connect(lambda : self.Upload_Imgs(2))
        self.RadioButton_SSD_2.clicked.connect(self.on_radio_selected)
        self.RadioButton_NND_2.clicked.connect(self.on_radio_selected)
        self.RadioButton_SIFT_2.clicked.connect(self.on_radio_selected)
        self.Slide_SIFT_Ratio.sliderReleased.connect(self.on_radio_selected)


        
       
        self.button_group = QButtonGroup(self)

       
        self.button_group.addButton(self.RadioButton_Harris_2)
        self.button_group.addButton(self.RadioButton_SIFT_2)
        self.button_group.addButton(self.RadioButton_Match_3)

       
        self.button_group.buttonClicked.connect(self.on_radio_selected)


       
        self.button_group = QButtonGroup(self)

 

        self.Label_Comp_Match.setText("     seconds")
        self.label_param_20.setText ("       Computation time :      seconds")
                # Create consistent colors using a fixed seed
       




    def k_value_changed(self):
        self.k_label.setText(str(f"K: {self.k_parameter.value()/100}"))

    def harris_thres_slider_value_changed(self):
        self.harris_thres_label.setText(str(f"Threshold: {self.harris_threshold.value() / 100}"))

    def lambda_thres_slider_value_changed(self):
        self.lambda_thres_label.setText(str(f"Threshold: {self.lambda_threshold.value() / 100}"))

    def lambda_kernel_value_changed(self):
        if self.lambda_kernel.value() == 1:
            k_size = 3
        elif self.lambda_kernel.value() == 2:
            k_size = 5
        else:
            k_size = 7
        self.lambda_kernel_label.setText(str(f"K_Size: {k_size}"))

    def harris_kernel_value_changed(self):
        if self.harris_kernel.value() == 1:
            k_size = 3
        elif self.harris_kernel.value() == 2:
            k_size = 5
        else:
            k_size = 7
        self.harris_kernel_label.setText(str(f"K_Size: {k_size}"))

    def mouseDoubleClickEvent(self, event):
        self.load_image()

    def load_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(None, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            if pixmap.isNull():
                print("Failed to load image for display")
                return
            self.original_cv_image = cv2.imread(self.image_path)
            self.grayscale_org_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2GRAY)
            self.height, self.width = self.grayscale_org_image.shape[:2]
            # self.original_cv_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if self.original_cv_image is None:
                print(f"Failed to load image with OpenCV: {self.image_path}")
                self.original_cv_image = None
                return
            scaled_pixmap = pixmap.scaled(
                self.original_image_label.width(),
                self.original_image_label.height(),
                Qt.KeepAspectRatio
            )
            self.original_image_label.setPixmap(scaled_pixmap)

    def activate_corner_detection(self):
        if self.corner_radio_btn.isChecked():
            self.harris_radio_btn.setEnabled(True)
            self.lambda_radio_btn.setEnabled(True)
        else:
            self.harris_radio_btn.setEnabled(False)
            self.lambda_radio_btn.setEnabled(False)

    def apply_corner_detection(self):
        if self.harris_radio_btn.isChecked():
            self.apply_harris_operator()
        elif self.lambda_radio_btn.isChecked():
            self.apply_lambda_minus()

    def apply_harris_operator(self):
        try:
            if not self.harris_radio_btn.isChecked() or self.original_cv_image is None:
                return

            smoothed_IxIx, smoothed_IyIy, smoothed_IxIy = self.get_derivatives_and_smooth(self.harris_kernel.value())
            R = np.zeros_like(self.grayscale_org_image, dtype=np.float32)
            k = self.k_parameter.value()/100
            print(f"K: {self.k_parameter.value() / 100}")
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    #     M = np.array([[smoothed_IxIx[i, j], smoothed_IxIy[i, j]],
                    #                  [smoothed_IxIy[i, j], smoothed_IyIy[i, j]]])
                    try:
                        determinant = ((smoothed_IxIx[i, j] * smoothed_IyIy[i, j]) -
                                       (smoothed_IxIy[i, j] * smoothed_IxIy[i, j]))
                        trace = smoothed_IxIx[i, j] + smoothed_IyIy[i, j]
                        R[i, j] = determinant - k * (trace ** 2)
                    except Exception as e:
                        print(f"Inner Try Statement: {e}")
            # R_normalized = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            threshold = (self.harris_threshold.value()/100) * R.max()
            print(f"Threshold: {self.harris_threshold.value()/100}")
            corner_mask = R > threshold
            corner_image = np.zeros_like(self.grayscale_org_image)
            corner_image[corner_mask] = 255

            pixmap = generate_heatmap_pixmap(R)
            self.output_image_label.setPixmap(pixmap)
            self.output_image_label.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print(e)

    def apply_lambda_minus(self):
        try:
            if not self.lambda_radio_btn.isChecked() or self.original_cv_image is None:
                return
            smoothed_IxIx, smoothed_IyIy, smoothed_IxIy = self.get_derivatives_and_smooth(self.lambda_kernel.value())

            R = np.zeros_like(self.grayscale_org_image, dtype=np.float32)

            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    M = np.array([[smoothed_IxIx[i, j], smoothed_IxIy[i, j]],
                                  [smoothed_IxIy[i, j], smoothed_IyIy[i, j]]])
                    eigenvalues = np.linalg.eigvals(M)
                    lambda_minus = min(eigenvalues)
                    R[i, j] = lambda_minus

            # R_normalized = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            threshold = 0.01 * R.max()
            corner_mask = R > threshold
            corner_image = np.zeros_like(self.grayscale_org_image)
            corner_image[corner_mask] = 255

            pixmap = generate_heatmap_pixmap(R)
            self.output_image_label.setPixmap(pixmap)
            self.output_image_label.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print(f"Lambda Minus Error: {e}")

    def get_derivatives_and_smooth(self, kernel):
        # Edge Detection
        sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        self.height, self.width = self.grayscale_org_image.shape[:2]
        Ix = np.zeros_like(self.grayscale_org_image, dtype=np.float32)
        Iy = np.zeros_like(self.grayscale_org_image, dtype=np.float32)
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                neighbors = self.grayscale_org_image[i - 1:i + 2, j - 1:j + 2]
                Ix[i, j] = np.sum(neighbors * sobel_x)
                Iy[i, j] = np.sum(neighbors * sobel_y)

        # Smoothing
        IxIx = Ix * Ix
        IyIy = Iy * Iy
        IxIy = Ix * Iy

        smoothed_IxIx = apply_smoothing_window(IxIx, self.height, self.width, kernel)
        smoothed_IyIy = apply_smoothing_window(IyIy, self.height, self.width, kernel)
        smoothed_IxIy = apply_smoothing_window(IxIy, self.height, self.width, kernel)
        return smoothed_IxIx, smoothed_IyIy, smoothed_IxIy


#//////////////////////////////////////////////MATCHING CODE////////////////////////////////////////////////////////////////////

    def Upload_Imgs(self, num_of_img):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(None, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not image_path:
            return

        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print(f"Failed to load image: {image_path}")
            return

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Initialize top and bottom images if they don't exist
        if not hasattr(self, 'top_image'):
            self.top_image = None
        if not hasattr(self, 'bottom_image'):
            self.bottom_image = None

        # Handle image upload scenarios
        if num_of_img == 1:  # Replace top image
            self.top_image = cv_image
        elif num_of_img == 2:  # Replace bottom image
            self.bottom_image = cv_image

        # After upload, ensure both images exist (create white image if needed)
        if self.top_image is None and self.bottom_image is not None:
            self.top_image = np.full((self.bottom_image.shape[0], self.bottom_image.shape[1], 3), 255, dtype=np.uint8)
        elif self.bottom_image is None and self.top_image is not None:
            self.bottom_image = np.full((self.top_image.shape[0], self.top_image.shape[1], 3), 255, dtype=np.uint8)
        elif self.top_image is None and self.bottom_image is None:
            return  # No images to display
        else:
            self.image, self.kp1,self.des1,self.template, self.kp2, self.des2 =self.get_keypoints_and_descriptors(self.top_image, self.bottom_image)
            self.upload_first=True
            self.on_radio_selected()

            

        # Ensure the widths match by resizing images to the smallest width
        target_width =self.original_image_label.width()
        
        # Calculate the target height for each image (equal height)
        # Get height of the display area
        display_height = self.original_image_label.height()
        # Calculate half height, accounting for the separator
        separator_height = 2  # Height of separator in pixels
        half_height = (display_height - separator_height) // 2
        
        # Resize both images to have the same width and equal heights
        resized_top = cv2.resize(self.top_image, (target_width, half_height))
        resized_bottom = cv2.resize(self.bottom_image, (target_width, half_height))
        
        # Create a separator (gray line)
        separator = np.full((separator_height, target_width, 3), 128, dtype=np.uint8)  # Gray color
        
        # Combine the top image, separator, and bottom image vertically
        combined_image = np.vstack((resized_top, separator, resized_bottom))

        # Convert the combined image to QImage for displaying
        height, width, channel = combined_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(combined_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale the pixmap to fit the label size while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.original_image_label.width(),
            self.original_image_label.height(),
            Qt.KeepAspectRatio
        )
        

        # Set the pixmap to the label
        self.original_image_label.setPixmap(scaled_pixmap)

    def extract_keypoints_octaves(self, image):
        height, width = image.shape
        image = image.astype(np.float32)
        num_octaves = int(np.log2(min(width, height))) - 3
        keypoints = []

        # Pre-allocate arrays for multiple octaves instead of using dictionaries
        # This avoids memory overhead from dictionary operations
        max_blurred_size = 6  # Maximum number of blurred images (5 + 1 for indexing from 1)
        max_dog_size = 5  # Maximum number of DoG images (4 + 1 for indexing from 1)

        grey_image = image  # No need to copy since we don't modify the original

        for octave in range(1, num_octaves + 1):
            if octave > 1:
                grey_image = cv2.resize(grey_image, (grey_image.shape[1] // 2, grey_image.shape[0] // 2),
                                        interpolation=cv2.INTER_NEAREST)

            # Current octave dimensions
            oct_height, oct_width = grey_image.shape

            # Pre-allocate arrays for this octave's images
            # Using 3D arrays instead of dictionaries saves memory and lookup time
            blurred_images = np.zeros((max_blurred_size, oct_height, oct_width), dtype=np.float32)
            DOG_images = np.zeros((max_dog_size, oct_height, oct_width), dtype=np.float32)

            # Create Gaussian pyramid
            k_base = math.sqrt(2)
            for i in range(1, 6):
                sigma = 1.0
                k_scale = pow(k_base, i - 1)
                blurred_images[i] = cv2.GaussianBlur(grey_image, (5, 5), sigma * k_scale)

            # Create DoG pyramid (reusing memory)
            for i in range(1, 5):
                DOG_images[i] = blurred_images[i + 1] - blurred_images[i]

            # Detect keypoints
            for i in range(2, 4):
                # Process interior pixels only (avoid edge checks in inner loop)
                for y in range(1, oct_height - 1):
                    for x in range(1, oct_width - 1):
                        center_val = DOG_images[i, y, x]

                        # Skip low contrast points early
                        if abs(center_val) < 0.03:
                            continue

                        # Check if it's an extremum
                        is_extremum = True
                        is_max = center_val > 0

                        # Check current, upper, and lower slices in one loop
                        for layer in range(i - 1, i + 2):
                            if not is_extremum:
                                break
                            for dy in range(-1, 2):
                                if not is_extremum:
                                    break
                                for dx in range(-1, 2):
                                    # Skip center pixel when on layer i
                                    if layer == i and dy == 0 and dx == 0:
                                        continue

                                    neighbor_val = DOG_images[layer, y + dy, x + dx]

                                    # Check if neighbor contradicts extremum status
                                    if is_max and neighbor_val >= center_val:
                                        is_extremum = False
                                        break
                                    elif not is_max and neighbor_val <= center_val:
                                        is_extremum = False
                                        break

                        if is_extremum and abs(center_val) > 0.03:  # Contrast threshold
                            # Apply subpixel localization using Taylor expansion
                            refined_pos = self.localize_extremum(DOG_images, i, y, x)

                            if refined_pos is not None:
                                dx, dy, ds = refined_pos

                                # Check if the refined keypoint passes edge test
                                if self.check_cornerness(DOG_images[i], y, x):
                                    # Calculate actual coordinates in original image
                                    original_x = (x + dx) * (2 ** (octave - 1))
                                    original_y = (y + dy) * (2 ** (octave - 1))

                                    # Store keypoint with minimal required data
                                    keypoint = {
                                        'x': original_x,
                                        'y': original_y,
                                        'octave': octave,
                                        'scale': i + ds,  # Include subpixel scale
                                        'response': abs(center_val),
                                        'source_image': blurred_images[i].copy()  # Only copy what's needed
                                    }
                                    keypoints.append(keypoint)

        # Process keypoints - assign orientation and compute descriptors
        self.keypoints = keypoints
        self.keypoints_with_orientation = self.assign_orientation(keypoints)
        self.descriptors = self.compute_descriptors(self.keypoints_with_orientation)

        return self.keypoints_with_orientation, self.descriptors

    def localize_extremum(self, dog_images, layer, y, x):
        """
        Refine keypoint location to subpixel accuracy using Taylor expansion.

        Args:
            dog_images: 3D array of Difference-of-Gaussian images
            layer: Current layer in DoG pyramid
            y, x: Pixel coordinates of the keypoint

        Returns:
            Tuple (dx, dy, ds) of subpixel refinement, or None if refinement failed
        """
        MAX_ITERATIONS = 5
        CONVERGENCE_THRESHOLD = 0.5  # Subpixel threshold

        # Derivative and Hessian matrix buffers (reused across iterations)
        derivative = np.zeros(3, dtype=np.float32)
        hessian = np.zeros((3, 3), dtype=np.float32)

        for _ in range(MAX_ITERATIONS):
            # Make sure we have enough pixels around for gradient calculation
            if layer <= 0 or layer >= len(dog_images) - 1 or y <= 1 or y >= dog_images[0].shape[0] - 2 or x <= 1 or x >= \
                    dog_images[0].shape[1] - 2:
                return None

            # Compute first derivatives (gradient)
            dx = (dog_images[layer, y, x + 1] - dog_images[layer, y, x - 1]) / 2.0
            dy = (dog_images[layer, y + 1, x] - dog_images[layer, y - 1, x]) / 2.0
            ds = (dog_images[layer + 1, y, x] - dog_images[layer - 1, y, x]) / 2.0
            derivative = np.array([dx, dy, ds])

            # Compute second derivatives (Hessian)
            dxx = dog_images[layer, y, x + 1] + dog_images[layer, y, x - 1] - 2 * dog_images[layer, y, x]
            dyy = dog_images[layer, y + 1, x] + dog_images[layer, y - 1, x] - 2 * dog_images[layer, y, x]
            dss = dog_images[layer + 1, y, x] + dog_images[layer - 1, y, x] - 2 * dog_images[layer, y, x]

            dxy = (dog_images[layer, y + 1, x + 1] - dog_images[layer, y + 1, x - 1] -
                   dog_images[layer, y - 1, x + 1] + dog_images[layer, y - 1, x - 1]) / 4.0
            dxs = (dog_images[layer + 1, y, x + 1] - dog_images[layer + 1, y, x - 1] -
                   dog_images[layer - 1, y, x + 1] + dog_images[layer - 1, y, x - 1]) / 4.0
            dys = (dog_images[layer + 1, y + 1, x] - dog_images[layer + 1, y - 1, x] -
                   dog_images[layer - 1, y + 1, x] + dog_images[layer - 1, y - 1, x]) / 4.0

            hessian = np.array([
                [dxx, dxy, dxs],
                [dxy, dyy, dys],
                [dxs, dys, dss]
            ])

            # Solve linear system for extremum offset using Hessian inverse * (-gradient)
            try:
                offset = np.linalg.solve(hessian, -derivative)
            except np.linalg.LinAlgError:
                # Hessian matrix is singular, skip this keypoint
                return None

            # If offset is small, we've converged
            if abs(offset[0]) < CONVERGENCE_THRESHOLD and abs(offset[1]) < CONVERGENCE_THRESHOLD and abs(
                    offset[2]) < CONVERGENCE_THRESHOLD:
                # Calculate extremum value
                extremum_value = dog_images[layer, y, x] + 0.5 * np.dot(derivative, offset)

                # Discard low contrast keypoints
                if abs(extremum_value) < 0.03:
                    return None

                # Return the refined subpixel position
                return offset

            # Update integer coordinates for next iteration
            x_new, y_new, layer_new = x + int(round(offset[0])), y + int(round(offset[1])), layer + int(
                round(offset[2]))

            # Check if new position is out of bounds
            if (layer_new < 1 or layer_new > len(dog_images) - 2 or
                    y_new < 1 or y_new > dog_images[0].shape[0] - 2 or
                    x_new < 1 or x_new > dog_images[0].shape[1] - 2):
                return None

            # Update for next iteration
            x, y, layer = x_new, y_new, layer_new

        # If we reached maximum iterations without convergence, still return the last offset
        # as long as it's valid (within bounds)
        if (0 <= offset[0] < 1.5 and 0 <= offset[1] < 1.5 and 0 <= offset[2] < 1.5):
            return offset
        return None

    def check_cornerness(self, image, y_coo, x_coo):
        edge_thresh = 10.0
        y = y_coo
        x = x_coo

        # Use second-order central differences
        Dxx = image[y, x + 1] + image[y, x - 1] - 2 * image[y, x]
        Dyy = image[y + 1, x] + image[y - 1, x] - 2 * image[y, x]
        Dxy = (image[y + 1, x + 1] - image[y + 1, x - 1] -
               image[y - 1, x + 1] + image[y - 1, x - 1]) / 4.0

        # Construct Hessian
        TrH = Dxx + Dyy  # Trace
        DetH = Dxx * Dyy - Dxy ** 2  # Determinant

        # Compute curvature ratio (avoid division by zero)
        if DetH <= 0:
            return False

        r = edge_thresh
        edge_pass = ((TrH ** 2) / DetH < ((r + 1) ** 2) / r)

        return edge_pass

    def assign_orientation(self, keypoints):
        """
        Assign orientation to keypoints based on local image gradient directions.
        Memory-optimized version.
        """
        keypoints_with_orientation = []

        # Pre-allocate histogram array to reuse for all keypoints
        orientation_histogram = np.zeros(36, dtype=np.float32)

        for keypoint in keypoints:
            x, y = int(keypoint['x']), int(keypoint['y'])
            octave = keypoint['octave']
            scale = keypoint['scale']
            source_image = keypoint['source_image']

            # Adjust for octave
            x_octave = x // (2 ** (octave - 1))
            y_octave = y // (2 ** (octave - 1))

            # Define window size
            window_radius = int(3 * 1.5 * scale)

            # Ensure window fits within image bounds
            height, width = source_image.shape

            # Reset histogram for this keypoint
            orientation_histogram.fill(0)

            # Process pixels in window
            for i in range(max(1, y_octave - window_radius), min(height - 2, y_octave + window_radius + 1)):
                for j in range(max(1, x_octave - window_radius), min(width - 2, x_octave + window_radius + 1)):
                    # Calculate gradient - with bounds checking
                    dx = source_image[i, min(width - 1, j + 1)] - source_image[i, max(0, j - 1)]
                    dy = source_image[min(height - 1, i + 1), j] - source_image[max(0, i - 1), j]

                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = (np.arctan2(dy, dx) + np.pi) * 180 / np.pi

                    # Apply Gaussian weight
                    dist_sq = ((i - y_octave) ** 2 + (j - x_octave) ** 2)
                    weight = np.exp(-dist_sq / (2 * (1.5 * scale) ** 2))
                    weighted_magnitude = weight * gradient_magnitude

                    # Add to histogram
                    bin_idx = int(gradient_orientation / 10) % 36  # 36 bins (each 10 degrees)
                    orientation_histogram[bin_idx] += weighted_magnitude

            # Smooth histogram efficiently in-place
            for _ in range(6):  # Smooth 6 times
                prev = orientation_histogram[35]
                for bin_idx in range(36):
                    current = orientation_histogram[bin_idx]
                    next_val = orientation_histogram[(bin_idx + 1) % 36]
                    orientation_histogram[bin_idx] = (prev + current + next_val) / 3.0
                    prev = current

            # Find peaks in histogram
            max_peak = np.max(orientation_histogram)
            peak_threshold = 0.8 * max_peak

            # Create keypoint for each significant orientation peak
            for bin_idx in range(36):
                if orientation_histogram[bin_idx] >= peak_threshold:
                    # Check if it's a local peak
                    prev_idx = (bin_idx - 1) % 36
                    next_idx = (bin_idx + 1) % 36
                    if (orientation_histogram[bin_idx] > orientation_histogram[prev_idx] and
                            orientation_histogram[bin_idx] > orientation_histogram[next_idx]):
                        # Refine peak by interpolation using quadratic fit
                        bin_center = bin_idx * 10

                        # Quadratic interpolation for more accurate peak
                        p1 = orientation_histogram[prev_idx]
                        p2 = orientation_histogram[bin_idx]
                        p3 = orientation_histogram[next_idx]

                        # Refined bin using quadratic interpolation formula
                        offset = 0.5 * (p1 - p3) / (p1 - 2 * p2 + p3)
                        if not np.isnan(offset) and -0.5 < offset < 0.5:
                            bin_center += offset * 10

                        orientation = bin_center % 360  # Ensure it's within 0-360 degrees

                        # Create new keypoint with this orientation
                        new_keypoint = keypoint.copy()
                        new_keypoint['orientation'] = orientation
                        keypoints_with_orientation.append(new_keypoint)

            # If no peaks found, use the max bin
            if not any(kp.get('x') == x and kp.get('y') == y for kp in keypoints_with_orientation):
                max_bin = np.argmax(orientation_histogram)
                bin_center = max_bin * 10 + 5
                new_keypoint = keypoint.copy()
                new_keypoint['orientation'] = bin_center
                keypoints_with_orientation.append(new_keypoint)

        return keypoints_with_orientation

    def compute_descriptors(self, keypoints_with_orientation):
        """
        Compute descriptor for each keypoint.
        Memory-optimized version.
        """
        descriptors = []
        num_bins = 8  # Number of bins per histogram
        descriptor_size = num_bins * 4 * 4  # 128 dimensions

        # Pre-allocate arrays
        descriptor = np.zeros(descriptor_size, dtype=np.float32)
        histogram = np.zeros(num_bins, dtype=np.float32)

        for keypoint in keypoints_with_orientation:
            x, y = int(keypoint['x']), int(keypoint['y'])
            octave = keypoint['octave']
            scale = keypoint['scale']
            orientation = keypoint['orientation']
            source_image = keypoint['source_image']

            # Convert orientation to radians
            orientation_rad = orientation * np.pi / 180.0

            # Adjust for octave
            x_octave = x // (2 ** (octave - 1))
            y_octave = y // (2 ** (octave - 1))

            # Reset descriptor array
            descriptor.fill(0)

            # Window size is determined by scale
            window_size = 3 * scale

            # Calculate cos and sin of orientation for rotation invariance
            cos_angle = np.cos(orientation_rad)
            sin_angle = np.sin(orientation_rad)

            # Calculate descriptor values
            height, width = source_image.shape
            hist_width = 4 * window_size
            half_width = hist_width / 2

            # Process each subregion more efficiently
            hist_index = 0
            for i in range(-2, 2):
                for j in range(-2, 2):
                    # Reset histogram for this subregion
                    histogram.fill(0)

                    # Center of subregion relative to keypoint
                    center_y = (i + 0.5) * window_size
                    center_x = (j + 0.5) * window_size

                    # Calculate window bounds
                    win_min = -int(window_size / 2)
                    win_max = int(window_size / 2) + 1

                    # For each pixel in subregion
                    for y_offset in range(win_min, win_max):
                        for x_offset in range(win_min, win_max):
                            # Rotate coordinates
                            rot_y = (y_offset * cos_angle - x_offset * sin_angle)
                            rot_x = (y_offset * sin_angle + x_offset * cos_angle)

                            # Position relative to subregion center
                            y_pos = rot_y + center_y
                            x_pos = rot_x + center_x

                            # Calculate contribution to histogram
                            y_bin = y_pos / half_width + 2
                            x_bin = x_pos / half_width + 2

                            # If sample is within descriptor bounds
                            if 0 <= y_bin < 4 and 0 <= x_bin < 4:
                                # Actual pixel coordinates (rotated around keypoint)
                                pixel_x = int(x_octave + (x_offset * cos_angle + y_offset * sin_angle))
                                pixel_y = int(y_octave + (-x_offset * sin_angle + y_offset * cos_angle))

                                # Check if we're within image bounds
                                if 0 <= pixel_y < height - 1 and 0 <= pixel_x < width - 1:
                                    # Calculate gradient
                                    dx = source_image[pixel_y, min(width - 1, pixel_x + 1)] - source_image[
                                        pixel_y, max(0, pixel_x - 1)]
                                    dy = source_image[min(height - 1, pixel_y + 1), pixel_x] - source_image[
                                        max(0, pixel_y - 1), pixel_x]

                                    # Gradient magnitude and orientation
                                    magnitude = np.sqrt(dx * dx + dy * dy)
                                    theta = (np.arctan2(dy, dx) + np.pi) * 180 / np.pi

                                    # Subtract keypoint orientation for rotation invariance
                                    theta = (theta - orientation) % 360

                                    # Weight by Gaussian
                                    weight = np.exp(-((rot_x * rot_x + rot_y * rot_y) / (2 * (0.5 * window_size) ** 2)))

                                    # Trilinear interpolation for histogram bin
                                    ori_bin = theta * num_bins / 360  # Convert to 0-8
                                    ori_bin_floor = int(np.floor(ori_bin))
                                    ori_bin_weight = ori_bin - ori_bin_floor

                                    # Make sure bin index is valid
                                    bin_idx = ori_bin_floor % num_bins

                                    # Add weighted contribution
                                    histogram[bin_idx] += weight * magnitude * (1 - ori_bin_weight)
                                    histogram[(bin_idx + 1) % num_bins] += weight * magnitude * ori_bin_weight

                    # Copy histogram to descriptor
                    descriptor[hist_index:hist_index + num_bins] = histogram
                    hist_index += num_bins

            # Normalize descriptor efficiently
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor /= norm

            # Clip values to 0.2 (as per SIFT paper)
            np.clip(descriptor, 0, 0.2, out=descriptor)

            # Normalize again
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor /= norm

            # Store descriptor with keypoint
            descriptors.append((keypoint, descriptor.copy()))

        return descriptors
    


    def match_descriptors(self, descriptors1, descriptors2, threshold=0.75):
        """
        Match descriptors between two images using ratio test.

        Args:
            descriptors1: List of (keypoint, descriptor) pairs from first image
            descriptors2: List of (keypoint, descriptor) pairs from second image
            threshold: Ratio threshold for Lowe's ratio test

        Returns:
            List of (keypoint1, keypoint2) matches
        """
        matches = []

        # For each descriptor in the first image
        for i, (kp1, desc1) in enumerate(descriptors1):
            # Find distances to all descriptors in the second image
            distances = []
            for j, (kp2, desc2) in enumerate(descriptors2):
                # Euclidean distance
                dist = np.linalg.norm(desc1 - desc2)
                distances.append((dist, j))

            # Sort by distance
            distances.sort()

            # If we have at least 2 matches
            if len(distances) >= 2:
                # Apply Lowe's ratio test
                if distances[0][0] < threshold * distances[1][0]:
                    best_match_idx = distances[0][1]
                    matches.append((kp1, descriptors2[best_match_idx][0]))

        return matches

    def visualize_keypoints(self, image, keypoints, color=(0, 255, 0)):
        """
        Visualize keypoints on the image.

        Args:
            image: Original image
            keypoints: List of keypoints
            color: Color for keypoint visualization

        Returns:
            Image with keypoints drawn
        """
        vis_image = image.copy()

        # Convert to color if grayscale
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        # Draw keypoints
        for keypoint in keypoints:
            x, y = int(keypoint['x']), int(keypoint['y'])

            # Draw circle at keypoint position
            cv2.circle(vis_image, (x, y), 3, color, -1)

            # Draw orientation line
            if 'orientation' in keypoint:
                orientation_rad = keypoint['orientation'] * np.pi / 180.0
                scale = 10  # Line length
                dx = int(scale * np.cos(orientation_rad))
                dy = int(scale * np.sin(orientation_rad))
                cv2.line(vis_image, (x, y), (x + dx, y + dy), color, 1)

        return vis_image

    def visualize_matches(self, image1, keypoints1, image2, keypoints2, matches):
        """
        Visualize matches between two images.

        Args:
            image1, image2: Original images
            keypoints1, keypoints2: List of keypoints from each image
            matches: List of (keypoint1, keypoint2) matches

        Returns:
            Combined image with match lines drawn
        """
        # Convert to color if grayscale
        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        # Create combined image
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        height = max(h1, h2)
        width = w1 + w2

        combined_image = np.zeros((height, width, 3), dtype=np.uint8)
        combined_image[:h1, :w1] = image1
        combined_image[:h2, w1:w1 + w2] = image2

        # Draw match lines
        for kp1, kp2 in matches:
            x1, y1 = int(kp1['x']), int(kp1['y'])
            x2, y2 = int(kp2['x']) + w1, int(kp2['y'])  # Adjust x2 for second image

            # Random color for each match
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Draw line
            cv2.line(combined_image, (x1, y1), (x2, y2), color, 1)

            # Draw points
            cv2.circle(combined_image, (x1, y1), 3, color, -1)
            cv2.circle(combined_image, (x2, y2), 3, color, -1)

        return combined_image

    def get_sift_keypoints_and_descriptors(self,img):
        """
        Wrapper function to make output compatible with OpenCV's SIFT format

        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: NumPy array of shape (N, 128) containing descriptors
        """
        # Call your existing function
        keypoints_with_orientation, descriptors_pairs = self.extract_keypoints_octaves(img)

        # Convert keypoints to cv2.KeyPoint objects
        cv_keypoints = []
        for kp in keypoints_with_orientation:
            # Create a cv2.KeyPoint object
            # Parameters: x, y, size, angle, response, octave, class_id
            cv_kp = cv2.KeyPoint(
                x=float(kp['x']),
                y=float(kp['y']),
                size=float(kp['scale'] * 3),  # Size based on scale
                angle=float(kp['orientation']),  # Orientation in degrees
                response=float(kp['response']),
                octave=int(kp['octave']),
                class_id=-1  # Default class_id
            )
            cv_keypoints.append(cv_kp)

        # Convert descriptors to numpy array
        cv_descriptors = np.array([desc for _, desc in descriptors_pairs])

        return cv_keypoints, cv_descriptors


  #hereeeeeee   
    def get_keypoints_and_descriptors(self, img, template  ):
        self.start_time_cal_sift = time.time()
        if img is None or template is None:
            print("One or both images failed to load.")
            return
                # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        kp1,des1=self.get_sift_keypoints_and_descriptors(img_gray)
        kp2, des2 = self.get_sift_keypoints_and_descriptors(template_gray)




        #hereeeeeee
        # sift = cv2.SIFT_create()
        # kp1_not, des1_not = sift.detectAndCompute(template_gray, None)
        # kp2_not, des2_not = sift.detectAndCompute(img_gray, None)
        # # Convert descriptors to float32


        # print(des2)
        des1 = np.array(des1).astype(np.float32)
        des2 = np.array(des2).astype(np.float32)
        self.elapsed_time_cal_sift = time.time() - self.start_time_cal_sift
        return img, kp1,des1,template, kp2, des2
   
   
   
    def sift_matching_drawing(self, img, kp1, desc1, template, kp2, desc2, ratio_thresh=0.75):
        # Start the timer
        start_time = time.time()
        
        # Initialize matches list
        matches = []
        
        # Pre-compute squared norms
        desc1_sq = np.sum(desc1 ** 2, axis=1)
        desc2_sq = np.sum(desc2 ** 2, axis=1)
        
        # Compute all pairwise distances using matrix operations
        dot_product = np.dot(desc1, desc2.T)
        ssd_matrix = desc1_sq[:, np.newaxis] + desc2_sq - 2 * dot_product
        
        # Find matches using ratio test
        for i in range(desc1.shape[0]):
            distances = ssd_matrix[i]
            smallest_indices = np.argpartition(distances, 1)[:2]
            dist1, dist2 = distances[smallest_indices[0]], distances[smallest_indices[1]]
            
            if dist1 > dist2:
                dist1, dist2 = dist2, dist1
                smallest_indices = smallest_indices[::-1]
            
            if dist1 < ratio_thresh * dist2:
                matches.append((i, smallest_indices[0], dist1))
        
        # Prepare images for visualization
        img1_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR) if len(template.shape) == 2 else template.copy()
        img2_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        
        # Create output image (side by side)
        h1, w1 = img1_color.shape[:2]
        h2, w2 = img2_color.shape[:2]
        output_img = np.ones((max(h1, h2), w1 + w2, 3), dtype=np.uint8) * 255  # White background
        output_img[:h1, :w1] = img1_color
        output_img[:h2, w1:w1+w2] = img2_color
        

        
        # Draw all matches at once (more efficient than drawing in the loop)
        for idx1, idx2, _ in matches:
            x1, y1 = int(kp1[idx1].pt[0]), int(kp1[idx1].pt[1])
            x2, y2 = int(kp2[idx2].pt[0] + w1), int(kp2[idx2].pt[1])
            
            # Generate consistent color for this match
            color = (255, 0,255)
            
            # Draw line and circles
            cv2.line(output_img, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
            cv2.circle(output_img, (x1, y1), 1, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(output_img, (x2, y2), 1, color, -1, lineType=cv2.LINE_AA)
        
        # Convert to QImage and display
        height, width, channel = output_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(output_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            472,
            self.output_image_label.height(),
            Qt.KeepAspectRatio
        )
        self.output_image_label.setPixmap(scaled_pixmap)
        
        # Calculate and display elapsed time
        if self.upload_first :
             self.upload_first=False
             print("////////////////////////////////////////")
             print(f"time sift :{self.elapsed_time_cal_sift}")
             print(f"time match  :{time.time() - start_time }")
            
             elapsed_time = time.time() - start_time + self.elapsed_time_cal_sift
        else:
                elapsed_time = time.time() - start_time
        self.label_param_20.setText(f"        Computation time : {elapsed_time:.4f} seconds")










    def sum_of_squared_difference(self, img, template):
        if img is None or template is None:
            print("One or both images failed to load.")
            return

        # Start the timer
        start_time = time.time()

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Get template dimensions
        temp_h, temp_w = template_gray.shape
        img_h, img_w = img_gray.shape

        min_ssd = float('inf')
        best_loc = (0, 0)

        vaild_h= img_h - temp_h + 1
        vaild_w= img_w - temp_w + 1

        # Manually compute SSD over all possible locations in the image
        for y in range(vaild_h):
            for x in range(vaild_w):
                # Extract the region of the image
                region = img_gray[y:y+temp_h, x:x+temp_w]
                
                # Compute SSD for this region and the template
                ssd = np.sum((region - template_gray) ** 2)

                # Track minimum SSD and best matching location
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_loc = (x, y)

        # Calculate similarity percentage from SSD
        max_possible_ssd = (255**2) * (temp_h * temp_w)
        similarity_percent = 100 * (1 - min_ssd / max_possible_ssd)

        # Draw rectangle on a copy of the original image
        output_image = img.copy()
        cv2.rectangle(output_image, best_loc, (best_loc[0] + temp_w, best_loc[1] + temp_h), (255, 0, 255), 1)

        # Convert to QImage and display
        height, width, channel = output_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(output_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
              472,
            self.output_image_label.height(),
            Qt.KeepAspectRatio
        )
        self.output_image_label.setPixmap(scaled_pixmap)

        # End the timer and calculate the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"SSD Similarity: {similarity_percent:.2f}%\n")
        self.Label_Comp_Match.setText(f"{elapsed_time:.3f} seconds")


    def normalized_cross_correlation(self, img, template):
        if img is None or template is None:
            print("One or both images failed to load.")
            return

        # Start the timer
        start_time = time.time()

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Get template dimensions
        temp_h, temp_w = template_gray.shape
        img_h, img_w = img_gray.shape

        max_ncc = -1.0
        best_loc = (0, 0)
        vaild_h= img_h - temp_h + 1
        vaild_w= img_w - temp_w + 1
        # Manually compute NCC over all possible locations in the image
        for y in range(vaild_h):
            for x in range(vaild_w):
                # Extract the region of the image
                region = img_gray[y:y+temp_h, x:x+temp_w]

                # Calculate the mean of the region and the template
                mean_img = np.mean(region)
                mean_template = np.mean(template_gray)

                # Compute the numerator and denominator of the NCC formula
                numerator = np.sum((region - mean_img) * (template_gray - mean_template))
                denominator = np.sqrt(np.sum((region - mean_img) ** 2) * np.sum((template_gray - mean_template) ** 2))

                # Compute NCC for this region
                ncc = numerator / denominator if denominator != 0 else 0

                # Track maximum NCC and best matching location
                if ncc > max_ncc:
                    max_ncc = ncc
                    best_loc = (x, y)

        # Calculate similarity percentage from NCC
        similarity_percent = 100 * max_ncc

        # Draw rectangle on a copy of the original image
        # Draw rectangle on a copy of the original image
        output_image = img.copy()
        cv2.rectangle(output_image, best_loc, (best_loc[0] + temp_w, best_loc[1] + temp_h), (0, 255, 255), 1)

        # Convert to QImage and display
        height, width, channel = output_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(output_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
             472,
            self.output_image_label.height(),
            Qt.KeepAspectRatio
        )
        self.output_image_label.setPixmap(scaled_pixmap)

        # End the timer and calculate the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"NCC Similarity: {similarity_percent:.2f}%\n")
        self.Label_Comp_Match.setText(f"{elapsed_time:.3f} seconds")




    def on_radio_selected(self):
        if  self.RadioButton_Harris_2.isChecked():
            pass
        elif  self.RadioButton_SIFT_2.isChecked():
                
                slider_val=self.Slide_SIFT_Ratio.value()
                self.label_param_19.setText(f"         SIFT_RATIO  : {slider_val/100:.2f}    ")
                if self.image is not  None:
                    self.sift_matching_drawing(self.image, self.kp1,self.des1,self.template, self.kp2, self.des2,ratio_thresh=slider_val/100 )
        
        elif self.RadioButton_Match_3.isChecked():
            if self.RadioButton_SSD_2.isChecked():
                self.sum_of_squared_difference(self.top_image, self.bottom_image)
            elif self.RadioButton_NND_2.isChecked():
                self.normalized_cross_correlation(self.top_image, self.bottom_image)
            





if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
