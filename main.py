import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load the image
image = cv2.imread("ivy.jpeg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the FAST detector object
# A lower threshold value will result in less suppression
fast = cv2.FastFeatureDetector_create(threshold=30)

# Detect keypoints using the FAST algorithm
keypoints = fast.detect(gray, None)

# Draw the keypoints on the image
result_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Display the original image with keypoints marked
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('FAST Feature Detection')
plt.show()