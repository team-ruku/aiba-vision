import argparse
import os
import torch
import cv2
import numpy as np
import networks
from torchvision.transforms import ToTensor
from layers import disp_to_depth
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model on a webcam input")
    parser.add_argument("--model_name", type=str, help="Name of the pre-trained model to test", required=True)
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA", default=False)
    return parser.parse_args()


def depth_color_map(input_image):
    cmap = plt.get_cmap("inferno")
    norm_image = cv2.normalize(input_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    colored_map = cmap(norm_image)[:, :, :3] * 255
    return cv2.cvtColor(colored_map.astype(np.float32), cv2.COLOR_RGB2BGR)


def run_webcam_example(args, encoder, depth_decoder):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Resize the webcam frame to match the model input size
        frame = cv2.resize(frame, (640, 192))

        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.cuda() if torch.cuda.is_available() else input_image

        with torch.no_grad():
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp_map = outputs[("disp", 0)].cpu().numpy().squeeze()

        depth_map = depth_color_map(disp_map)
        
        # Print min and max depth values
        print(f"Min value: {np.min(disp_map)}, Max value: {np.max(disp_map)}")

        # Display depth map with colorbar
        plt.figure()
        plt.imshow(depth_map)
        plt.colorbar()
        plt.show()
        
        # Convert frame and depth_map to float32 before merging
        frame = frame.astype(np.float32)
        depth_map = depth_map.astype(np.float32)
        merged_view = cv2.addWeighted(frame, 0.5, depth_map, 0.5, 0)
        
        # Resize the merged view to fit the screen
        merged_view = cv2.resize(merged_view, (640, 480))
        cv2.imshow("Webcam Output", cv2.cvtColor(merged_view, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def main():
    args = parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = os.path.join("models", args.model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)

    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    run_webcam_example(args, encoder, depth_decoder)


if __name__ == "__main__":
    main()
