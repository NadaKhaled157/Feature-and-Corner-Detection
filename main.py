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


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
