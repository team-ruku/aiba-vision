import cv2
import numpy as np

# Read the depth map and point cloud data from the files.
depth_map = cv2.imread("depth_map.png", cv2.IMREAD_GRAYSCALE)
point_cloud = np.load("point_cloud.npy")

# Convert the depth map to a point cloud.
points = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
points[:, :, 0] = depth_map
points[:, :, 1] = depth_map
points[:, :, 2] = depth_map

# Visualize the point cloud.
cv2.imshow("Point cloud", points)
cv2.waitKey(0)