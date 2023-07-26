import numpy as np
import cv2
import open3d as o3d

def disparity_to_depth(disparity_map, focal_length, baseline):
    depth_map = (focal_length * baseline) / disparity_map
    return depth_map

def generate_point_cloud(depth_map, focal_length, cx, cy):
    height, width = depth_map.shape
    points = []

    for v in range(height):
        for u in range(width):
            depth = depth_map[v, u]
            if depth > 0:
                x = (u - cx) * depth / focal_length
                y = (v - cy) * depth / focal_length
                z = depth
                points.append([x, y, z])

    return np.array(points)

def main():
    # Disparity map 파일 로드
    disparity_map = cv2.imread('test_image_disp.jpeg', cv2.IMREAD_GRAYSCALE)

    # Camera parameters 설정 (이 값들은 실제 카메라의 내부 파라미터에 맞게 설정해야 합니다.)
    focal_length = 500  # Focal length (f)
    baseline = 0.1      # Stereo camera baseline (T)

    # Disparity map을 Depth map으로 변환
    depth_map = np.where(disparity_map > 0, (focal_length * baseline) / disparity_map, 0)


    # Point Cloud 생성
    cx, cy = disparity_map.shape[1] / 2, disparity_map.shape[0] / 2  # 이미지의 중심점 (Principal point)
    point_cloud = generate_point_cloud(depth_map, focal_length, cx, cy)

    # Point Cloud 시각화
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
