import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def depth_map_to_point_cloud(depth_map, focal_length, principal_point, scaling_factor=1.0):
    height, width = depth_map.shape

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    mesh_x, mesh_y = np.meshgrid(x, y)

    x3D = (mesh_x - principal_point[0]) * depth_map / focal_length[0] / scaling_factor
    y3D = (mesh_y - principal_point[1]) * depth_map / focal_length[1] / scaling_factor
    z3D = depth_map

    point_cloud = np.dstack((x3D, y3D, z3D))

    return point_cloud

def depth_map_to_color_map(depth_map, color_map='viridis'):
    depth_map_normalized = depth_map / np.max(depth_map)
    color_map_output = plt.get_cmap(color_map)(depth_map_normalized)[:, :, :3]
    color_map_output = (color_map_output * 255).astype(np.uint8)

    return color_map_output

def visualize_and_save_point_cloud(point_cloud, output_file):
    x = point_cloud[:, :, 0].flatten()
    y = point_cloud[:, :, 1].flatten()
    z = point_cloud[:, :, 2].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0, antialiased=True, s=1)  # reduce the point size with s

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(output_file)
    plt.close(fig)

# Load depth map image
depth_map = cv2.imread("depth_map.png", cv2.IMREAD_ANYDEPTH)

# Set camera intrinsics (example values)
focal_length = (700, 700)    
principal_point = (320, 240) 
scaling_factor = 1.0

# Convert depth map to point cloud
point_cloud = depth_map_to_point_cloud(depth_map, focal_length, principal_point, scaling_factor)

# Set output file name
output_file = "point_cloud_image.png"

# Visualize and save the point cloud as an image
visualize_and_save_point_cloud(point_cloud, output_file)

# Convert depth map to color map
color_map_output = depth_map_to_color_map(depth_map)

# Save color map image
cv2.imwrite("color_map_output.png", color_map_output)
