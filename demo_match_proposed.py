import torch
import numpy as np
import kornia as K
from PIL import Image
import cv2
from utils.test_utils import detect
from model.network import RLFDB
from third_party.hardnet.hardnet_pytorch import HardNet
import argparse
import os
import time

def load_im(im_path):
    im = Image.open(im_path)
    im_rgb = im.convert('RGB')
    im_gray = im_rgb.convert('L')
    im_rgb = np.array(im_rgb)
    im_gray = np.array(im_gray)
    return im_rgb, im_gray

def extract_features(args, im_rgb, im_gray, detector, descriptor, device):
    kpts_np = detect(args, im_rgb, detector, device)
    num_kpts = kpts_np.shape[0]
    
    if num_kpts == 0:
        return np.zeros([0,2]), np.array([])
    
    kpts = torch.from_numpy(kpts_np[:, 0:2])  # Only use x, y coordinates
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
    start_time = time.time()
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

    # Load images
    print("Loading images...")
    im_rgb1, im_gray1 = load_im('assets/mffw_06_A.jpg')
    im_rgb2, im_gray2 = load_im('assets/mffw_06_B.jpg')

    # Keypoint detection stage
    print("\nDetecting keypoints...")
    detection_start = time.time()
    kpts1 = detect(args, im_rgb1, detector, device)
    kpts2 = detect(args, im_rgb2, detector, device)
    detection_end = time.time()
    detection_time = detection_end - detection_start
    print(f"Keypoint detection completed: {detection_time:.3f} seconds")
    print(f"Image 1: {len(kpts1)} keypoints, Image 2: {len(kpts2)} keypoints")
    
    # Feature description stage
    print("\nExtracting descriptors...")
    descriptor_start = time.time()
    
    # Extract descriptors for the first image
    if len(kpts1) > 0:
        kpts_tensor1 = torch.from_numpy(kpts1[:, 0:2])  # Only use x, y coordinates
        kp1 = torch.cat([kpts_tensor1[:, 0].view(-1, 1).float(), kpts_tensor1[:, 1].view(-1, 1).float()],dim=1).unsqueeze(0).to(device)
        s1 = args.s_mult * torch.ones((1, len(kpts1), 1, 1)).to(device)
        src_laf1 = K.feature.laf_from_center_scale_ori(kp1, s1, torch.zeros((1, len(kpts1), 1)).to(device))
        timg_gray1 = torch.from_numpy(im_gray1).unsqueeze(0).unsqueeze(0).to(device)
        patches1 = K.feature.extract_patches_from_pyramid(timg_gray1.float() / 255.0, src_laf1, PS=32)[0]
        desc1 = descriptor(patches1.to(device)).cpu().detach().numpy()
    else:
        desc1 = np.array([])
    
    # Extract descriptors for the second image
    if len(kpts2) > 0:
        kpts_tensor2 = torch.from_numpy(kpts2[:, 0:2])  # Only use x, y coordinates
        kp2 = torch.cat([kpts_tensor2[:, 0].view(-1, 1).float(), kpts_tensor2[:, 1].view(-1, 1).float()],dim=1).unsqueeze(0).to(device)
        s2 = args.s_mult * torch.ones((1, len(kpts2), 1, 1)).to(device)
        src_laf2 = K.feature.laf_from_center_scale_ori(kp2, s2, torch.zeros((1, len(kpts2), 1)).to(device))
        timg_gray2 = torch.from_numpy(im_gray2).unsqueeze(0).unsqueeze(0).to(device)
        patches2 = K.feature.extract_patches_from_pyramid(timg_gray2.float() / 255.0, src_laf2, PS=32)[0]
        desc2 = descriptor(patches2.to(device)).cpu().detach().numpy()
    else:
        desc2 = np.array([])
    
    descriptor_end = time.time()
    descriptor_time = descriptor_end - descriptor_start
    print(f"Descriptor extraction completed: {descriptor_time:.3f} seconds")
    
    # Feature matching stage
    print("\nMatching features...")
    matching_start = time.time()
    
    if desc1.size == 0 or desc2.size == 0:
        matches1 = np.zeros([0,2])
        matches2 = np.zeros([0,2])
    else:
        with torch.inference_mode():
            _, match_ids = K.feature.match_smnn(
                torch.from_numpy(desc1), torch.from_numpy(desc2), 0.99
            )
        
        if match_ids.size(0) == 0:
            matches1 = np.zeros([0,2])
            matches2 = np.zeros([0,2])
        else:
            matches1 = kpts1[match_ids[:, 0], :2]  # Only use x, y coordinates
            matches2 = kpts2[match_ids[:, 1], :2]  # Only use x, y coordinates
    
    matching_end = time.time()
    matching_time = matching_end - matching_start
    print(f"Feature matching completed: {matching_time:.3f} seconds")

    # Save matching results
    output_path = r"results/matches_MFFW_6.png"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    if matches1.shape[0] > 0 and matches2.shape[0] > 0:
        result_img = draw_matches(im_rgb1, matches1, im_rgb2, matches2, show_num=100)
        Image.fromarray(result_img).save(output_path)
        
        # Calculate runtime and display information
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n=== Feature Matching Completed ===")
        print(f"Number of matches found: {matches1.shape[0]}")
        print(f"\n--- Timing Breakdown ---")
        print(f"Keypoint detection: {detection_time:.3f} seconds")
        print(f"Descriptor extraction: {descriptor_time:.3f} seconds")
        print(f"Feature matching: {matching_time:.3f} seconds")
        print(f"Total runtime: {runtime:.3f} seconds")
        print(f"\nResult saved to: {os.path.abspath(output_path)}")
        print(f"Device used: {device}")
    else:
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n=== Feature Matching Completed ===")
        print(f"No matches found")
        print(f"\n--- Timing Breakdown ---")
        print(f"Keypoint detection: {detection_time:.3f} seconds")
        print(f"Descriptor extraction: {descriptor_time:.3f} seconds")
        print(f"Feature matching: {matching_time:.3f} seconds")
        print(f"Total runtime: {runtime:.3f} seconds")
        print(f"Device used: {device}")