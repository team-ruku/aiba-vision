import numpy as np

# Disparity map를 읽어온다.
disparity_map = np.load("/Users/antonio/Desktop/python project opt/monodepth2/test_image_disp.npy")

# 카메라 내부 파라미터 (실제 값으로 설정해야 함)
focal_length = 1000  # 카메라의 초점거리 (pixels)

# Depth map을 계산한다.
depth_map = focal_length / disparity_map

# Depth map을 저장한다.
np.save("/Users/antonio/Desktop/python project opt/monodepth2/test_image_dep.npy", depth_map)
