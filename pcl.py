import cv2
import numpy as np

# Read the depth map image.
depth_map = cv2.imread('depth_map.png', cv2.IMREAD_UNCHANGED)

# Convert the depth map to a point cloud.
points = np.array([[x, y, depth] for x, y, depth in np.ndenumerate(depth_map)])

# Save the point cloud.
np.save('point_cloud.npy', points)