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


def generate_harris_heatmap_pixmap(R, threshold_ratio=0.01):
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
#         height, width = cv_image.shape
#         bytes_per_line = width
#         qimage = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
#     elif len(cv_image.shape) == 3:  # Color image
#         height, width, channels = cv_image.shape
#         if channels == 3:
#             # Convert BGR to RGB
#             cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
#             bytes_per_line = channels * width
#             qimage = QImage(cv_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
#         else:
#             raise ValueError(f"Unsupported number of channels: {channels}")
#     else:
#         raise ValueError("Unsupported image format")
#     return QPixmap.fromImage(qimage)


def apply_smoothing_window(component):
    kernel = np.ones((3, 3)) / 9
    height, width = component.shape[:2]
    smoothed = np.copy(component)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighbors = component[i - 1:i + 2, j - 1:j + 2]
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
        self.original_image_label = self.findChild(QLabel, "Widget_Org_Image")
        self.original_image_label.setScaledContents(True)
        self.output_image_label = self.findChild(QLabel, "Widget_Output_1")
        self.harris_radio_btn = self.findChild(QRadioButton, "RadioButton_Harris")
        self.harris_radio_btn.toggled.connect(self.apply_harris_operator)

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
            # self.original_cv_image = cv2.imread(self.image_path)
            self.original_cv_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if self.original_cv_image is None:
                print(f"Failed to load image with OpenCV: {self.image_path}")
                self.original_cv_image = None
                return
            self.original_image_label.setPixmap(pixmap)

    def apply_harris_operator(self):
        if not self.harris_radio_btn.isChecked() or self.original_cv_image is None:
            return
        if check_if_grayscale(self.original_cv_image):
            print("Grayscale")
            derivatives = self.get_derivatives()
            try:
                Ix = derivatives["Ix"]
                Iy = derivatives["Iy"]
                IxIx = Ix * Ix
                IyIy = Iy * Iy
                IxIy = Ix * Iy
                smoothed_IxIx = apply_smoothing_window(IxIx)
                smoothed_IyIy = apply_smoothing_window(IyIy)
                smoothed_IxIy = apply_smoothing_window(IxIy)
            except Exception as e:
                print(f"Outer Try Statement: {e}")
            height, width = self.original_cv_image.shape[:2]
            try:
                R = np.zeros_like(self.original_cv_image, dtype=np.float32)
            except Exception as e:
                print(f"IDK: {e}")
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    #     M = np.array([[smoothed_IxIx[i, j], smoothed_IxIy[i, j]],
                    #                  [smoothed_IxIy[i, j], smoothed_IyIy[i, j]]])
                    try:
                        determinant = ((smoothed_IxIx[i, j] * smoothed_IyIy[i, j]) -
                                       (smoothed_IxIy[i, j] * smoothed_IxIy[i, j]))
                        trace = smoothed_IxIx[i, j] + smoothed_IyIy[i, j]
                        k = 0.04
                        R[i, j] = determinant - k * (trace ** 2)
                    except Exception as e:
                        print(f"Inner Try Statement: {e}")
            R_normalized = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            threshold = 0.01 * R.max()
            corner_mask = R > threshold
            corner_image = np.zeros_like(self.original_cv_image)
            corner_image[corner_mask] = 255

            pixmap = generate_harris_heatmap_pixmap(R)
            self.output_image_label.setPixmap(pixmap)
            self.output_image_label.setAlignment(Qt.AlignCenter)
        else:
            print("RGB")

    def get_derivatives(self):
        sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        height, width = self.original_cv_image.shape[:2]
        Ix = np.zeros_like(self.original_cv_image, dtype=np.float32)
        Iy = np.zeros_like(self.original_cv_image, dtype=np.float32)
        # # Convert image to grayscale if it's color
        # if len(self.original_cv_image.shape) == 3:
        #     gray_img = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray_img = self.original_cv_image
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighbors = self.original_cv_image[i - 1:i + 2, j - 1:j + 2]
                Ix[i, j] = np.sum(neighbors * sobel_x)
                Iy[i, j] = np.sum(neighbors * sobel_y)
        return {
            'Ix': Ix,
            'Iy': Iy,
        }


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
