import os
import torch
import numpy as np
from PIL import Image
import cv2
from utils import test_utils
from model.network import RLFDB
import argparse

def load_im(im_path):
    im = Image.open(im_path)
    im_rgb = im.convert('RGB')
    im_rgb = np.array(im_rgb)
    return im_rgb

def detect_and_save(args, im, detector, device):
    im = im / 255.
    height_RGB_norm, width_RGB_norm = im.shape[0], im.shape[1]
    image_even = test_utils.make_shape_even(im)
    height_even, width_even = image_even.shape[0], image_even.shape[1]
    image_pad = test_utils.mod_padding_symmetric(image_even, factor=64)
    image_pad_tensor = torch.tensor(image_pad, dtype=torch.float32)
    image_pad_tensor = image_pad_tensor.permute(2, 0, 1)
    image_pad_batch = image_pad_tensor.unsqueeze(0)

    with torch.inference_mode():
        output_pad_batch = detector(image_pad_batch.to(device))

    score_map_pad_batch = output_pad_batch['prob']
    score_map_pad_np = score_map_pad_batch[0, :, :].detach().cpu().numpy()

    # unpad images
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]
    score_map_remove_border = test_utils.remove_borders(score_map, borders=args.border_size)
    pts = test_utils.get_points_direct_from_score_map(
        heatmap=score_map_remove_border,
        conf_thresh=args.heatmap_confidence_threshold,
        nms_size=args.nms_size,
        subpixel=args.sub_pixel,
        patch_size=args.patch_size,
        order_coord=args.order_coord
    )
    if pts.size == 0:
        return np.zeros([0, 4])
    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:args.num_features]
    return pts_output

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
    args = parse_test_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = RLFDB(dims=[32, 64, 128, 256], layers=[2, 2, 6, 2], expand_ratio=3.0, mlp_ratio=3.0, use_dw=True,
                     drop_path_rate=0.05).to(device)
    checkpoint = torch.load(args.ckpt_file, map_location=torch.device('cpu') if device=='cpu' else None, weights_only=False)
    model_state_disk = checkpoint['model_state']
    detector.load_state_dict(model_state_disk)
    detector = detector.eval()

    im_rgb = load_im(args.input_image)
    keypoints = detect_and_save(args, im_rgb, detector, device)
    img_vis = np.array(im_rgb).copy()
    for pt in keypoints:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_vis, (x, y), 3, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite(args.output_image, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()