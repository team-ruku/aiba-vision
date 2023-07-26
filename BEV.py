import cv2
import numpy as np
from mayavi import mlab

def depth_map_to_point_cloud(depth_map, focal_length, principal_point, scaling_factor=1.0, downsampling_rate=5):
    height, width = depth_map.shape
    x = np.linspace(0, width - 1, width)[::downsampling_rate]
    y = np.linspace(0, height - 1, height)[::downsampling_rate]
    mesh_x, mesh_y = np.meshgrid(x, y)
    x3D = (mesh_x - principal_point[0]) * depth_map[::downsampling_rate, ::downsampling_rate] / focal_length[0] / scaling_factor
    y3D = (mesh_y - principal_point[1]) * depth_map[::downsampling_rate, ::downsampling_rate] / focal_length[1] / scaling_factor
    z3D = depth_map[::downsampling_rate, ::downsampling_rate]
    point_cloud = np.dstack((x3D, y3D, z3D))
    return point_cloud

def visualize_and_save_point_cloud_mesh_bird_eye_view(point_cloud, output_file, representation):
    x = point_cloud[:, :, 0].flatten()
    y = point_cloud[:, :, 1].flatten()
    z = point_cloud[:, :, 2].flatten()
    pts = mlab.points3d(x, y, z, z, scale_mode='none', scale_factor=1)
    mesh = mlab.pipeline.delaunay2d(pts)
    surf = mlab.pipeline.surface(mesh, representation=representation)
    mlab.view(azimuth=0, elevation=90)
    mlab.savefig(output_file)
    mlab.close()

# Load depth map image
depth_map = cv2.imread("depth_map.png", cv2.IMREAD_ANYDEPTH)

# Set camera intrinsics (example values)
focal_length = (700, 700)    
principal_point = (320, 240) 
scaling_factor = 1.0
downsampling_rate = 5

# Convert depth map to point cloud
point_cloud = depth_map_to_point_cloud(depth_map, focal_length, principal_point, scaling_factor, downsampling_rate)

# Set output file name for the mesh
output_file_bird_eye_view_mesh = "bird_eye_view_mesh.png"
representation = "wireframe"   # 'surface' for solid mesh, 'wireframe' for wireframe mesh

# Visualize and save the point cloud as a mesh in bird's eye view
visualize_and_save_point_cloud_mesh_bird_eye_view(point_cloud, output_file_bird_eye_view_mesh, representation)
