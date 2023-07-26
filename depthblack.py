import argparse
import os
import torch

assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

from torch.autograd import Variable
from torch.backends import cudnn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from models import DisparityNet

# 평가 코드 함수
def evaluate(cap, disp_net, output_directory, output_name):
    disp_net.eval()
    cudnn.benchmark = True

    # 텐서 생성
    tensor = np.random.randn(2, 3) 

    # 입력 이미지 임시 통합 후 텐서 집어넣기
    input_images = torch.from_numpy(tensor).float()
    input_images = input_images.cuda()

    # 시맨틱 분할 및 깊이 맵 생성
    input_images = Variable(input_images)
    with torch.no_grad():
        disp_resized = disp_net(input_images)

    # 그레이스케일 깊이 이미지 저장
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    disp_resized_np = (disp_resized_np - disp_resized_np.min()) / (disp_resized_np.max() - disp_resized_np.min())
    grayscale_im = (disp_resized_np * 255).astype(np.uint8)
    im = pil.fromarray(grayscale_im)

    # 결과물 파일 이름 변경 (.png)
    name_dest_im = os.path.join(output_directory, "{}_disp.png".format(output_name))

    # 결과 영상 저장
    im.save(name_dest_im)

    print("result image has been saved in {}".format(name_dest_im))


if __name__ == '__main__':
    # 모델 경로 바꾸기
    dispnet_model_path = "/path/to/your/dispnet_checkpoint.pth"

    # 결과물 디렉토리 설정
    output_directory = "/path/to/output/directory"
    output_name = "example_result"

    # 모델 불러오기
    disp_net = DisparityNet()
    weights = torch.load(dispnet_model_path)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.cuda()

    # 웹캠 실행 (미리 생성해두셨다고 가정)
    cap = cv2.VideoCapture(0)

    # 평가 함수 호출
    evaluate(cap, disp_net, output_directory, output_name)

    # 웹캠 종료 (미리 생성해두셨다고 가정)
    cap.release()
    cv2.destroyAllWindows()
