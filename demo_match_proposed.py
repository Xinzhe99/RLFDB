import torch
import numpy as np
import kornia as K
from PIL import Image
import cv2
from utils import test_utils
from model.network import RLFDB
from third_party.hardnet.hardnet_pytorch import HardNet
import argparse

def load_im(im_path):
    im = Image.open(im_path)
    im_rgb = im.convert('RGB')
    im_gray = im_rgb.convert('L')
    im_rgb = np.array(im_rgb)
    im_gray = np.array(im_gray)
    return im_rgb, im_gray

def detect(args, im, detector, device):
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

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]
    score_map_remove_border = test_utils.remove_borders(score_map, borders=args.border_size)
    pts = test_utils.get_points_direct_from_score_map(
        heatmap=score_map_remove_border, conf_thresh=args.heatmap_confidence_threshold,
        nms_size=args.nms_size, subpixel=args.sub_pixel,
        patch_size=args.patch_size, order_coord=args.order_coord
    )
    
    if pts.size == 0:
        return np.zeros([0,3])

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:args.num_features]

    return pts_output[:, 0:3]

def extract_features(args, im_rgb, im_gray, detector, descriptor, device):
    kpts_np = detect(args, im_rgb, detector, device)
    num_kpts = kpts_np.shape[0]
    
    if num_kpts == 0:
        return np.zeros([0,2]), np.array([])
    
    kpts = torch.from_numpy(kpts_np)
    kp = torch.cat([kpts[:, 0].view(-1, 1).float(), kpts[:, 1].view(-1, 1).float()],dim=1).unsqueeze(0).to(device)
    s = args.s_mult * torch.ones((1, num_kpts, 1, 1)).to(device)
    src_laf = K.feature.laf_from_center_scale_ori(kp, s, torch.zeros((1, num_kpts, 1)).to(device))
    timg_gray = torch.from_numpy(im_gray).unsqueeze(0).unsqueeze(0).to(device)
    patches = K.feature.extract_patches_from_pyramid(timg_gray.float() / 255.0, src_laf, PS=32)[0]
    
    if len(patches) > 1000:
        descs = None
        for i_patches in range(len(patches) // 1000 + 1):
            start_idx = 1000 * i_patches
            end_idx = 1000 * (i_patches + 1)
            patch_slice = patches[start_idx:end_idx]
            
            if patch_slice.size(0) == 0:
                continue

            descs_tmp = descriptor(patch_slice.to(device))

            if descs is None:
                descs = descs_tmp
            else:
                descs = torch.cat([descs, descs_tmp], dim=0)

        if descs is not None:
            descs = descs.cpu().detach().numpy()
        else:
            descs = np.array([])
    else:
        descs = descriptor(patches.to(device)).cpu().detach().numpy()
    return kpts_np[:, 0:2], descs

def extract_matches(
        args,
        im_rgb1, im_gray1, im_rgb2, im_gray2,
        detector, descriptor, device):
    kpts1, desc1 = extract_features(args, im_rgb1, im_gray1, detector, descriptor, device)
    kpts2, desc2 = extract_features(args, im_rgb2, im_gray2, detector, descriptor, device)

    if desc1.size == 0 or desc2.size == 0:
        return np.zeros([0,2]), np.zeros([0,2])

    # K.feature.match_smnn better
    with torch.inference_mode():
        _, match_ids = K.feature.match_smnn(
            torch.from_numpy(desc1), torch.from_numpy(desc2), 0.99
        )

    if match_ids.size(0) == 0:
        return np.zeros([0,2]), np.zeros([0,2])

    points1 = kpts1[match_ids[:, 0], :2]
    points2 = kpts2[match_ids[:, 1], :2]
    return points1, points2

def draw_matches(im1, kpts1, im2, kpts2, show_num=None):
    """所有连线和点都为绿色且加粗"""
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = im1
    out_img[:h2, w1:w1 + w2] = im2

    if show_num is not None:
        num = min(show_num, len(kpts1), len(kpts2))
    else:
        num = min(len(kpts1), len(kpts2))
    color = (0, 255, 0)  # 绿色，BGR
    for i in range(num):
        x1, y1 = kpts1[i]
        x2, y2 = kpts2[i]
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2 + w1)), int(round(y2)))
        cv2.line(out_img, pt1, pt2, color, 2, cv2.LINE_AA)
        cv2.circle(out_img, pt1, 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out_img, pt2, 4, color, -1, lineType=cv2.LINE_AA)
    return out_img

def parse_test_config():
    parser = argparse.ArgumentParser(description='Feature matching with RLFDB detector and HardNet descriptor')
    parser.add_argument('--ckpt_detector_file', type=str, 
                       default=r'pretrained/checkpoint.pth',
                       help='Path to detector checkpoint file.')
    parser.add_argument('--ckpt_descriptor_file', type=str,
                       default='third_party/hardnet/HardNet++.pth',
                       help='Path to HardNet descriptor checkpoint file.')
    parser.add_argument('--border_size', type=int, default=15)
    parser.add_argument('--nms_size', type=int, default=15)
    parser.add_argument('--num_features', type=int, default=2048)
    parser.add_argument('--s_mult', type=int, default=60)
    parser.add_argument('--order_coord', type=str, default='xysr')
    parser.add_argument('--heatmap_confidence_threshold', type=float, default=0.001)
    parser.add_argument('--sub_pixel', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=4)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_test_config()

    detector = RLFDB(dims=[32, 64, 128, 256], layers=[2, 2, 6, 2], expand_ratio=3.0,
                     mlp_ratio=3.0, use_dw=True, drop_path_rate=0.05).to(device)
    checkpoint_detector = torch.load(args.ckpt_detector_file, 
                                   map_location=torch.device('cpu') if device=='cpu' else None, 
                                   weights_only=False)
    model_state_disk = checkpoint_detector['model_state']
    detector.load_state_dict(model_state_disk)
    detector = detector.eval()

    # 加载HardNet描述符
    descriptor = HardNet()
    checkpoint_descriptor = torch.load(args.ckpt_descriptor_file, weights_only=True)
    descriptor.load_state_dict(checkpoint_descriptor['state_dict'])
    descriptor = descriptor.eval().to(device)

    # 加载图像
    im_rgb1, im_gray1 = load_im('assets/mffw_06_A.jpg')
    im_rgb2, im_gray2 = load_im('assets/mffw_06_B.jpg')

    # 提取匹配
    matches1, matches2 = extract_matches(
        args,
        im_rgb1, im_gray1, im_rgb2, im_gray2,
        detector, descriptor, device)

    # 保存匹配结果
    if matches1.shape[0] > 0 and matches2.shape[0] > 0:
        result_img = draw_matches(im_rgb1, matches1, im_rgb2, matches2, show_num=100)
        Image.fromarray(result_img).save(r"results/matches_MFFW_6.png")
        print(f"成功找到 {matches1.shape[0]} 个匹配点")
    else:
        print("未找到匹配点")