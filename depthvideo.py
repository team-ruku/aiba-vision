import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import cv2

import torch
from torchvision import transforms

import networks
from layers import disp_to_depth

def parse_args():
    parser = argparse.ArgumentParser(
        description='Monodepthv2 video testing function.')

    parser.add_argument('--video_path', type=str,
                        help='path to the input video file', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

def test_video(args):
    """Function to predict depth maps for video frames
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # Load pretrained model
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # Extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # Open the video file
    cap = cv2.VideoCapture(args.video_path)

    # Loop over video frames
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Prepare the input frame
            input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = pil.fromarray(input_image)
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # Prediction
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)].squeeze().cpu().numpy()

            # Resize and display disparity
            disp_resized = cv2.resize(disp, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            disp_color = cv2.applyColorMap((disp_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            blended = cv2.addWeighted(frame, 0.6, disp_color, 0.4, 0)
            cv2.imshow('Blended', blended)

            # Press 'q' to quit video playback and processing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video read and write objects and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    test_video(args)
