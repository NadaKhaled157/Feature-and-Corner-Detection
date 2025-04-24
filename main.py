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
        """
        Extract SIFT keypoints across multiple octaves, following Lowe's paper more precisely.
        """
        # Initialize parameters following the SIFT paper
        sigma = 1.6  # Initial sigma
        assumed_blur = 0.5  # Assumed initial blur in the image
        intervals = 3  # Number of intervals per octave (s)
        num_scales = intervals + 3  # Number of scales per octave
        k = 2 ** (1.0 / intervals)  # Scale factor between consecutive scales

        # Convert image to float
        image = image.astype(np.float32)

        # Pre-blur the image to achieve initial sigma
        if assumed_blur != sigma:
            sigma_diff = np.sqrt(sigma ** 2 - assumed_blur ** 2)
            image = cv2.GaussianBlur(image, (0, 0), sigma_diff)

        # Determine number of octaves based on image size
        height, width = image.shape
        num_octaves = int(np.log2(min(width, height))) - 3

        # Prepare data structures for keypoints
        all_keypoints = []

        # Process each octave
        for octave in range(num_octaves):
            # Create scale space for this octave
            gaussian_images = self.generate_gaussian_pyramid(image, num_scales, sigma, k)
            dog_images = self.generate_dog_pyramid(gaussian_images)

            # Find keypoint candidates
            keypoint_candidates = self.find_scale_space_extrema(dog_images, intervals, octave)

            # Refine keypoints with subpixel accuracy
            refined_keypoints = []
            for keypoint in keypoint_candidates:
                refined = self.refine_keypoint(dog_images, keypoint)
                if refined:
                    refined_keypoints.append(refined)

            # Add to all keypoints
            all_keypoints.extend(refined_keypoints)

            # Prepare base image for next octave (downsample)
            if octave < num_octaves - 1:
                image = cv2.resize(gaussian_images[-3], (gaussian_images[-3].shape[1] // 2,
                                                         gaussian_images[-3].shape[0] // 2),
                                   interpolation=cv2.INTER_NEAREST)

        # Assign orientations to keypoints
        keypoints_with_orientation = self.assign_orientation(all_keypoints)

        # Compute descriptors
        descriptors = self.compute_descriptors(keypoints_with_orientation)

        self.keypoints = all_keypoints
        self.keypoints_with_orientation = keypoints_with_orientation

        return keypoints_with_orientation, descriptors

    def generate_gaussian_pyramid(self, base_image, num_scales, sigma, k):
        """
        Generate Gaussian pyramid with proper scaling as per Lowe's paper.
        """
        height, width = base_image.shape
        gaussian_images = [base_image]

        # For each scale level
        for i in range(1, num_scales):
            sigma_current = sigma * (k ** (i - 1))
            sigma_diff = sigma_current * np.sqrt(k ** 2 - 1)

            # Blur the previous image to get the next scale
            blurred_image = cv2.GaussianBlur(gaussian_images[-1], (0, 0), sigma_diff)
            gaussian_images.append(blurred_image)

        return gaussian_images

    def generate_dog_pyramid(self, gaussian_images):
        """
        Generate Difference-of-Gaussian pyramid from Gaussian pyramid.
        """
        dog_images = []

        # Compute differences between adjacent scales
        for i in range(1, len(gaussian_images)):
            dog_images.append(gaussian_images[i] - gaussian_images[i - 1])

        return dog_images

    def find_scale_space_extrema(self, dog_images, intervals, octave):
        """
        Find extrema points in the DoG pyramid.
        """
        keypoints = []
        dog_height, dog_width = dog_images[0].shape
        contrast_threshold = 0.03  # As used in Lowe's paper

        # For each candidate keypoint location
        for scale_idx in range(1, len(dog_images) - 1):
            for y in range(1, dog_height - 1):
                for x in range(1, dog_width - 1):
                    # Check if this point is an extrema
                    if self.is_extremum(dog_images, scale_idx, y, x):
                        # Store candidate keypoint for refinement
                        keypoints.append({
                            'octave': octave,
                            'scale_idx': scale_idx,
                            'x': x,
                            'y': y,
                            'scale': intervals + scale_idx,  # Convert to actual scale
                        })

        return keypoints

    def is_extremum(self, dog_images, scale_idx, y, x):
        """
        Check if a point is an extremum in 3x3x3 neighborhood.
        """
        center_value = dog_images[scale_idx][y, x]

        # Low contrast threshold check (early rejection)
        if abs(center_value) < 0.03:
            return False

        # Check if it's a maximum
        if center_value > 0:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        # Skip the center
                        if i == 0 and j == 0 and k == 0:
                            continue

                        # Check if any neighbor is greater
                        if dog_images[scale_idx + i][y + j, x + k] >= center_value:
                            return False
            return True

        # Check if it's a minimum
        if center_value < 0:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        # Skip the center
                        if i == 0 and j == 0 and k == 0:
                            continue

                        # Check if any neighbor is smaller
                        if dog_images[scale_idx + i][y + j, x + k] <= center_value:
                            return False
            return True

        return False

    def refine_keypoint(self, dog_images, keypoint):
        """
        Refine keypoint location to subpixel accuracy using 3D quadratic fit.
        """
        MAX_ITER = 5
        CONVERGENCE_THRESHOLD = 0.5

        octave = keypoint['octave']
        scale_idx = keypoint['scale_idx']
        x = keypoint['x']
        y = keypoint['y']

        # Fit 3D quadratic and check for convergence
        for _ in range(MAX_ITER):
            # Extract 3D slice around the point
            dog_slice = [
                [dog_images[scale_idx - 1][y - 1:y + 2, x - 1:x + 2]],
                [dog_images[scale_idx][y - 1:y + 2, x - 1:x + 2]],
                [dog_images[scale_idx + 1][y - 1:y + 2, x - 1:x + 2]]
            ]

            # Calculate derivative
            dx = (dog_images[scale_idx][y, x + 1] - dog_images[scale_idx][y, x - 1]) / 2.0
            dy = (dog_images[scale_idx][y + 1, x] - dog_images[scale_idx][y - 1, x]) / 2.0
            ds = (dog_images[scale_idx + 1][y, x] - dog_images[scale_idx - 1][y, x]) / 2.0
            gradient = np.array([dx, dy, ds])

            # Calculate Hessian matrix
            dxx = dog_images[scale_idx][y, x + 1] + dog_images[scale_idx][y, x - 1] - 2 * dog_images[scale_idx][y, x]
            dyy = dog_images[scale_idx][y + 1, x] + dog_images[scale_idx][y - 1, x] - 2 * dog_images[scale_idx][y, x]
            dss = dog_images[scale_idx + 1][y, x] + dog_images[scale_idx - 1][y, x] - 2 * dog_images[scale_idx][y, x]

            dxy = (dog_images[scale_idx][y + 1, x + 1] - dog_images[scale_idx][y + 1, x - 1] -
                   dog_images[scale_idx][y - 1, x + 1] + dog_images[scale_idx][y - 1, x - 1]) / 4.0

            dxs = (dog_images[scale_idx + 1][y, x + 1] - dog_images[scale_idx + 1][y, x - 1] -
                   dog_images[scale_idx - 1][y, x + 1] + dog_images[scale_idx - 1][y, x - 1]) / 4.0

            dys = (dog_images[scale_idx + 1][y + 1, x] - dog_images[scale_idx + 1][y - 1, x] -
                   dog_images[scale_idx - 1][y + 1, x] + dog_images[scale_idx - 1][y - 1, x]) / 4.0

            hessian = np.array([
                [dxx, dxy, dxs],
                [dxy, dyy, dys],
                [dxs, dys, dss]
            ])

            # Solve for offset
            try:
                offset = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                # Matrix is singular, keypoint is unstable
                return None

            # Check for convergence
            if abs(offset[0]) < CONVERGENCE_THRESHOLD and abs(offset[1]) < CONVERGENCE_THRESHOLD and abs(
                    offset[2]) < CONVERGENCE_THRESHOLD:
                break

            # Update position
            x += int(round(offset[0]))
            y += int(round(offset[1]))
            scale_idx += int(round(offset[2]))

            # Check if updated position is still valid
            if (scale_idx < 1 or scale_idx > len(dog_images) - 2 or
                    y < 1 or y > dog_images[0].shape[0] - 2 or
                    x < 1 or x > dog_images[0].shape[1] - 2):
                return None

        # Check contrast after refinement (Lowe's criterion)
        value = dog_images[scale_idx][y, x] + 0.5 * np.dot(gradient, offset)
        if abs(value) < 0.03:
            return None

        # Check edge response using Hessian
        if not self.check_edge_response(hessian):
            return None

        # Calculate final keypoint properties
        kp = {
            'x': (x + offset[0]) * (2 ** octave),  # Map back to original image coords
            'y': (y + offset[1]) * (2 ** octave),
            'octave': octave,
            'scale': scale_idx + offset[2],
            'response': abs(value),
            'source_image': dog_images[scale_idx]  # Keep for orientation assignment
        }

        return kp

    def check_edge_response(self, hessian):
        """
        Check edge response using eigenvalue ratio (as in Lowe's paper).
        """
        edge_threshold = 10.0  # As used in Lowe's paper

        # Extract principal curvatures from 2x2 part of Hessian
        trace = hessian[0, 0] + hessian[1, 1]
        det = hessian[0, 0] * hessian[1, 1] - hessian[0, 1] * hessian[1, 0]

        # Avoid division by zero
        if det <= 0:
            return False

        # Calculate eigenvalue ratio
        r = trace ** 2 / det
        r_threshold = (edge_threshold + 1) ** 2 / edge_threshold

        return r < r_threshold

    def assign_orientation(self, keypoints):
        """
        Assign orientation to keypoints as described in Lowe's paper.
        """
        keypoints_with_orientation = []

        for keypoint in keypoints:
            x, y = keypoint['x'], keypoint['y']
            octave = keypoint['octave']
            scale = keypoint['scale']
            source_image = keypoint['source_image']

            # Adjust coordinates to current octave
            x_octave = int(x / (2 ** octave))
            y_octave = int(y / (2 ** octave))

            # Calculate the window radius for gradient analysis (as per Lowe's paper)
            sigma = 1.5 * scale
            radius = int(3 * sigma)

            # Create orientation histogram (36 bins for 360 degrees)
            hist = np.zeros(36)

            # Image dimensions
            height, width = source_image.shape

            # Process pixels in window
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    yi = y_octave + i
                    xi = x_octave + j

                    # Skip pixels outside the image
                    if yi < 1 or yi >= height - 1 or xi < 1 or xi >= width - 1:
                        continue

                    # Calculate gradient
                    dx = source_image[yi, xi + 1] - source_image[yi, xi - 1]
                    dy = source_image[yi + 1, xi] - source_image[yi - 1, xi]

                    magnitude = np.sqrt(dx * dx + dy * dy)
                    orientation = (np.arctan2(dy, dx) + np.pi) * 180.0 / np.pi  # 0-360 degrees

                    # Calculate bin index
                    bin_idx = int(np.floor(orientation * 36 / 360))
                    bin_idx = bin_idx % 36  # Ensure it's in range [0,35]

                    # Apply Gaussian weight
                    weight = np.exp(-(i * i + j * j) / (2 * sigma * sigma))

                    # Add to histogram
                    hist[bin_idx] += weight * magnitude

            # Smooth histogram 6 times as in the paper
            for _ in range(6):
                hist_smooth = np.copy(hist)
                for bin_idx in range(36):
                    prev_idx = (bin_idx - 1) % 36
                    next_idx = (bin_idx + 1) % 36
                    hist_smooth[bin_idx] = (hist[prev_idx] + hist[bin_idx] + hist[next_idx]) / 3.0
                hist = hist_smooth

            # Find peaks and create keypoints
            hist_max = np.max(hist)
            threshold = 0.8 * hist_max

            # For each peak above threshold
            for bin_idx in range(36):
                if hist[bin_idx] >= threshold:
                    # Check if it's a local peak
                    prev_idx = (bin_idx - 1) % 36
                    next_idx = (bin_idx + 1) % 36

                    if hist[bin_idx] > hist[prev_idx] and hist[bin_idx] > hist[next_idx]:
                        # Interpolate peak position for better accuracy
                        bin_center = bin_idx

                        # Fit parabola to 3 points
                        p1 = hist[prev_idx]
                        p2 = hist[bin_idx]
                        p3 = hist[next_idx]

                        # Quadratic peak interpolation
                        if p1 != p2 and p3 != p2:
                            offset = 0.5 * (p1 - p3) / (p1 - 2 * p2 + p3)
                            if -0.5 < offset < 0.5:  # Valid interpolation
                                bin_center += offset

                        # Convert to degrees
                        orientation = (bin_center * 10) % 360

                        # Create keypoint with orientation
                        new_keypoint = keypoint.copy()
                        new_keypoint['orientation'] = orientation
                        keypoints_with_orientation.append(new_keypoint)

            # If no peaks were found, use the max bin
            if not any(kp['x'] == x and kp['y'] == y for kp in keypoints_with_orientation):
                max_idx = np.argmax(hist)
                orientation = (max_idx * 10 + 5) % 360
                new_keypoint = keypoint.copy()
                new_keypoint['orientation'] = orientation
                keypoints_with_orientation.append(new_keypoint)

        return keypoints_with_orientation

    def compute_descriptors(self, keypoints_with_orientation):
        """
        Compute SIFT descriptors as described in Lowe's paper.
        """
        descriptors = []

        for keypoint in keypoints_with_orientation:
            x, y = keypoint['x'], keypoint['y']
            octave = keypoint['octave']
            scale = keypoint['scale']
            orientation = keypoint['orientation']
            source_image = keypoint['source_image']

            # Convert to radians
            orientation_rad = orientation * np.pi / 180.0

            # Adjust to octave coordinates
            x_octave = int(x / (2 ** octave))
            y_octave = int(y / (2 ** octave))

            # Descriptor parameters
            num_bins = 8  # Histogram bins per direction
            num_regions = 4  # 4x4 descriptor grid
            descriptor_width = num_regions * num_bins  # 128 dimensions total

            # Calculate cos and sin for rotation
            cos_angle = np.cos(orientation_rad)
            sin_angle = np.sin(orientation_rad)

            # Size of descriptor region relative to keypoint scale
            descriptor_sigma = 0.5 * num_regions
            descriptor_radius = int(np.sqrt(2) * descriptor_sigma * scale * 3)

            # Initialize descriptor array
            descriptor = np.zeros((num_regions, num_regions, num_bins), dtype=np.float32)

            # Process pixels in window
            height, width = source_image.shape

            for i in range(-descriptor_radius, descriptor_radius + 1):
                for j in range(-descriptor_radius, descriptor_radius + 1):
                    # Rotate coordinates
                    rot_j = j * cos_angle - i * sin_angle
                    rot_i = j * sin_angle + i * cos_angle

                    # Position in descriptor array
                    region_x = (rot_j / scale) / descriptor_sigma + (num_regions - 1) / 2
                    region_y = (rot_i / scale) / descriptor_sigma + (num_regions - 1) / 2

                    # Skip if outside descriptor region
                    if region_x < -0.5 or region_x > num_regions - 0.5 or region_y < -0.5 or region_y > num_regions - 0.5:
                        continue

                    # Get image coordinates
                    img_y = y_octave + i
                    img_x = x_octave + j

                    # Skip if outside image bounds
                    if img_y < 1 or img_y >= height - 1 or img_x < 1 or img_x >= width - 1:
                        continue

                    # Calculate gradient
                    dx = source_image[img_y, img_x + 1] - source_image[img_y, img_x - 1]
                    dy = source_image[img_y + 1, img_x] - source_image[img_y - 1, img_x]

                    # Rotate gradient to keypoint orientation
                    rot_dx = dx * cos_angle + dy * sin_angle
                    rot_dy = -dx * sin_angle + dy * cos_angle

                    # Magnitude and orientation
                    magnitude = np.sqrt(rot_dx * rot_dx + rot_dy * rot_dy)
                    theta = (np.arctan2(rot_dy, rot_dx) + np.pi) * 180.0 / np.pi  # 0-360 degrees

                    # Convert orientation to 0-8 bins
                    ori_bin = theta * num_bins / 360.0

                    # Weight with Gaussian centered on descriptor
                    weight = np.exp(-(rot_j * rot_j + rot_i * rot_i) / (2 * (descriptor_sigma * scale) ** 2))

                    # Trilinear interpolation
                    for region_y_idx in range(2):
                        y_idx = int(np.floor(region_y)) + region_y_idx
                        if y_idx < 0 or y_idx >= num_regions:
                            continue

                        y_weight = 1.0 - abs(region_y - y_idx)

                        for region_x_idx in range(2):
                            x_idx = int(np.floor(region_x)) + region_x_idx
                            if x_idx < 0 or x_idx >= num_regions:
                                continue

                            x_weight = 1.0 - abs(region_x - x_idx)

                            # Orientation bin interpolation
                            ori_idx = int(np.floor(ori_bin))
                            ori_frac = ori_bin - ori_idx

                            # Wrap around for histogram
                            ori_idx = ori_idx % num_bins

                            # Add weighted contribution to descriptor
                            w = weight * x_weight * y_weight
                            descriptor[y_idx, x_idx, ori_idx] += magnitude * w * (1 - ori_frac)
                            descriptor[y_idx, x_idx, (ori_idx + 1) % num_bins] += magnitude * w * ori_frac

            # Flatten descriptor to 1D array
            flat_descriptor = descriptor.flatten()

            # Normalize descriptor for illumination invariance
            norm = np.linalg.norm(flat_descriptor)
            if norm > 0:
                flat_descriptor /= norm

            # Clip values to 0.2 (as in Lowe's paper)
            flat_descriptor = np.minimum(flat_descriptor, 0.2)

            # Normalize again
            norm = np.linalg.norm(flat_descriptor)
            if norm > 0:
                flat_descriptor /= norm

            descriptors.append((keypoint, flat_descriptor))

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
   
   
   
    def sift_matching_drawing(self, template, kp1, desc1, img, kp2, desc2, ratio_thresh=0.75):
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
