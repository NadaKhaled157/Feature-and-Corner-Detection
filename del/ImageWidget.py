# from PyQt5.QtWidgets import *
# from PyQt5.QtGui import *
#
#
# class ImageWidget(QLabel):
#     def __init__(self):
#         super().__init__()
#         self.setScaledContents(True)
#
#     def mouseDoubleClickEvent(self, event):
#         file_dialog = QFileDialog()
#         image_path, _ = file_dialog.getOpenFileNames(None, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
#         if image_path:
#             pixmap = QPixmap(image_path)
#             self.setPixmap(pixmap)
#