import os
import torch
import numpy as np
from PIL import Image
import cv2
from utils.test_utils import detect
from model.network import RLFDB
import argparse
import time

def load_im(im_path):
    im = Image.open(im_path)
    im_rgb = im.convert('RGB')
    im_rgb = np.array(im_rgb)
    return im_rgb

# detect_and_save function is now imported from test_utils

def parse_test_config():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--ckpt_file', type=str, default='pretrained/checkpoint.pth', help='Path to checkpoint file.')
    parser.add_argument('--input_image', type=str, default='assets/kitti_selected_motion_blur_median.png', help='Path to input image.')
    parser.add_argument('--output_image', type=str, default='results/kitti_result.png', help='Path to save output image.')
    parser.add_argument('--border_size', type=int, default=15)
    parser.add_argument('--nms_size', type=int, default=3)
    parser.add_argument('--num_features', type=int, default=2048)
    parser.add_argument('--s_mult', type=int, default=60)
    parser.add_argument('--order_coord', type=str, default='xysr')
    parser.add_argument('--heatmap_confidence_threshold', type=float, default=0.001)
    parser.add_argument('--sub_pixel', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=4)
    args = parser.parse_args()
    return args

def main():
    start_time = time.time()
    args = parse_test_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = RLFDB(dims=[32, 64, 128, 256], layers=[2, 2, 6, 2], expand_ratio=3.0, mlp_ratio=3.0, use_dw=True,
                     drop_path_rate=0.05).to(device)
    checkpoint = torch.load(args.ckpt_file, map_location=torch.device('cpu') if device=='cpu' else None, weights_only=False)
    model_state_disk = checkpoint['model_state']
    detector.load_state_dict(model_state_disk)
    detector = detector.eval()

    im_rgb = load_im(args.input_image)
    keypoints = detect(args, im_rgb, detector, device)
    img_vis = np.array(im_rgb).copy()
    for pt in keypoints:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_vis, (x, y), 3, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Save the image
    cv2.imwrite(args.output_image, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    
    # Calculate runtime and display information
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n=== Execution Completed ===")
    print(f"Number of keypoints detected: {len(keypoints)}")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Result saved to: {os.path.abspath(args.output_image)}")
    print(f"Device used: {device}")

if __name__ == "__main__":
    main()