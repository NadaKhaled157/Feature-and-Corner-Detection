import cv2
import numpy as np

# Load the image
image = cv2.imread("Corners.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Convert to float32 as required by the Harris function
gray = np.float32(gray)

# Apply the Harris Corner Detector
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

# Dilate the result for better visualization
dst = cv2.dilate(dst, None)

# Threshold for detecting corners
image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red

# Display the result
cv2.imshow("Harris Corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()