import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 4: Load the image
img = cv2.imread('ATU.jpeg')
if img is None:
    raise FileNotFoundError("Image file 'ATU.jpeg' not found.")

# Step 5: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 6: Plot original and grayscale images
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 1, 2), plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Step 7: Apply Gaussian Blur with different kernel sizes
blurred_5x5 = cv2.GaussianBlur(gray, (5, 5), 0)
blurred_9x9 = cv2.GaussianBlur(gray, (9, 9), 0)

# Plot Gaussian Blur results
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1), plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(blurred_5x5, cmap='gray')
plt.title('5x5 Gaussian Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(blurred_9x9, cmap='gray')
plt.title('9x9 Gaussian Blur'), plt.xticks([]), plt.yticks([])
plt.show()

# Step 8: Sobel Edge Detection
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges

# Plot Sobel results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

# Step 9: Combine Sobel results
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Plot combined Sobel edges
plt.figure(figsize=(5, 5))
plt.imshow(sobel_combined, cmap='gray')
plt.title('Combined Sobel Edges')
plt.xticks([]), plt.yticks([])
plt.show()

# Step 10: Canny Edge Detection
canny_edges = cv2.Canny(gray, 100, 200)

# Plot Canny edges
plt.figure(figsize=(5, 5))
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.xticks([]), plt.yticks([])
plt.show()

# Advanced Task: Thresholding Sobel Results
threshold = 100
sobel_thresholded = np.where(sobel_combined > threshold, 1, 0)

# Plot thresholded Sobel results
plt.figure(figsize=(5, 5))
plt.imshow(sobel_thresholded, cmap='gray')
plt.title('Thresholded Sobel Edges')
plt.xticks([]), plt.yticks([])
plt.show()
