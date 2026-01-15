"""
Simple Edge Detection Comparison
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create synthetic image (512x512, grayscale)
img = np.zeros((512, 512), dtype=np.uint8)
img[:] = 50  # Background intensity

# Draw filled square
cv2.rectangle(img, (100, 100), (250, 250), 200, -1)

# Draw filled circle
cv2.circle(img, (350, 350), 80, 150, -1)

# Save original
cv2.imwrite('/Users/eli/synthetic_image.png', img)

# Create ground truth edges (we know exactly where they are)
ground_truth = np.zeros((512, 512), dtype=np.uint8)
# Square edges
cv2.rectangle(ground_truth, (100, 100), (250, 250), 255, 2)
# Circle edge
cv2.circle(ground_truth, (350, 350), 80, 255, 2)

print("="*60)
print("EDGE DETECTION COMPARISON")
print("="*60)

# Test 1: Clean image, default thresholds
print("\n[Test 1] Clean image with default thresholds")
print("-"*60)

canny = cv2.Canny(img, 50, 150)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = np.uint8(sobel / sobel.max() * 255)
_, sobel = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = np.uint8(np.abs(laplacian) / np.abs(laplacian).max() * 255)
_, laplacian = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)

# Simple metric: count edge pixels
gt_pixels = np.sum(ground_truth > 0)
canny_pixels = np.sum(canny > 0)
sobel_pixels = np.sum(sobel > 0)
lap_pixels = np.sum(laplacian > 0)

print(f"Ground truth edge pixels: {gt_pixels}")
print(f"Canny detected:           {canny_pixels} ({canny_pixels/gt_pixels:.2f}x)")
print(f"Sobel detected:           {sobel_pixels} ({sobel_pixels/gt_pixels:.2f}x)")
print(f"Laplacian detected:       {lap_pixels} ({lap_pixels/gt_pixels:.2f}x)")

# Visualize Test 1
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(ground_truth, cmap='gray')
axes[0, 1].set_title('Ground Truth Edges')
axes[0, 1].axis('off')

axes[0, 2].imshow(canny, cmap='gray')
axes[0, 2].set_title('Canny')
axes[0, 2].axis('off')

axes[1, 0].imshow(sobel, cmap='gray')
axes[1, 0].set_title('Sobel')
axes[1, 0].axis('off')

axes[1, 1].imshow(laplacian, cmap='gray')
axes[1, 1].set_title('Laplacian')
axes[1, 1].axis('off')

axes[1, 2].axis('off')

plt.suptitle('Test 1: Clean Image, Default Thresholds')
plt.tight_layout()
plt.savefig('/Users/eli/test1_clean.png', dpi=150)
print("Saved: test1_clean.png")

# Test 2: Add noise
print("\n[Test 2] With Gaussian noise")
print("-"*60)

noise = np.random.normal(0, 25, img.shape)
noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

canny2 = cv2.Canny(noisy_img, 50, 150)
sobel_x2 = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y2 = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=3)
sobel2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
sobel2 = np.uint8(sobel2 / sobel2.max() * 255)
_, sobel2 = cv2.threshold(sobel2, 50, 255, cv2.THRESH_BINARY)

laplacian2 = cv2.Laplacian(noisy_img, cv2.CV_64F)
laplacian2 = np.uint8(np.abs(laplacian2) / np.abs(laplacian2).max() * 255)
_, laplacian2 = cv2.threshold(laplacian2, 30, 255, cv2.THRESH_BINARY)

canny2_pixels = np.sum(canny2 > 0)
sobel2_pixels = np.sum(sobel2 > 0)
lap2_pixels = np.sum(laplacian2 > 0)

print(f"Canny detected:           {canny2_pixels} ({canny2_pixels/gt_pixels:.2f}x)")
print(f"Sobel detected:           {sobel2_pixels} ({sobel2_pixels/gt_pixels:.2f}x)")
print(f"Laplacian detected:       {lap2_pixels} ({lap2_pixels/gt_pixels:.2f}x)")

# Visualize Test 2
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(noisy_img, cmap='gray')
axes[0, 0].set_title('Noisy Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(canny2, cmap='gray')
axes[0, 1].set_title('Canny')
axes[0, 1].axis('off')

axes[1, 0].imshow(sobel2, cmap='gray')
axes[1, 0].set_title('Sobel')
axes[1, 0].axis('off')

axes[1, 1].imshow(laplacian2, cmap='gray')
axes[1, 1].set_title('Laplacian')
axes[1, 1].axis('off')

plt.suptitle('Test 2: Noisy Image (sigma=25)')
plt.tight_layout()
plt.savefig('/Users/eli/test2_noisy.png', dpi=150)
print("Saved: test2_noisy.png")

# Test 3: Different thresholds
print("\n[Test 3] Clean image with higher thresholds")
print("-"*60)

canny3 = cv2.Canny(img, 100, 200)
_, sobel3 = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)
_, laplacian3 = cv2.threshold(laplacian, 60, 255, cv2.THRESH_BINARY)

canny3_pixels = np.sum(canny3 > 0)
sobel3_pixels = np.sum(sobel3 > 0)
lap3_pixels = np.sum(laplacian3 > 0)

print(f"Canny detected:           {canny3_pixels} ({canny3_pixels/gt_pixels:.2f}x)")
print(f"Sobel detected:           {sobel3_pixels} ({sobel3_pixels/gt_pixels:.2f}x)")
print(f"Laplacian detected:       {lap3_pixels} ({lap3_pixels/gt_pixels:.2f}x)")

# Visualize Test 3
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(canny3, cmap='gray')
axes[0].set_title('Canny (100, 200)')
axes[0].axis('off')

axes[1].imshow(sobel3, cmap='gray')
axes[1].set_title('Sobel (thresh=100)')
axes[1].axis('off')

axes[2].imshow(laplacian3, cmap='gray')
axes[2].set_title('Laplacian (thresh=60)')
axes[2].axis('off')

plt.suptitle('Test 3: Higher Thresholds')
plt.tight_layout()
plt.savefig('/Users/eli/test3_high_thresh.png', dpi=150)
print("Saved: test3_high_thresh.png")

print("\n" + "="*60)
print("Done! Check the saved PNG files.")
print("="*60)

plt.show()
