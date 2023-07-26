import cv2
import numpy as np

# Read the input disparity map.
disparity_map = cv2.imread('test2_image_disp.jpeg', cv2.IMREAD_GRAYSCALE)

# Convert the disparity map to a depth map.
depth_map = cv2.convertScaleAbs(disparity_map, alpha=1.0 / 16.0)

# Save the output depth map.
cv2.imwrite('depth_map2.png', depth_map)