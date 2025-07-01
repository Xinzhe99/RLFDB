import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import test_utils
from model.network import RLFDB
import argparse

#未完成
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
    return pts_output  # 返回[N,4]的数组

def detect_proposed(input_root,output_root):
    def parse_test_config():
        parser = argparse.ArgumentParser(description='eval')

        parser.add_argument('--ckpt_file', type=str,
                            default=r'pretrained/checkpoint.pth',
                            help='The path to the checkpoint file to load the detector weights.')
        parser.add_argument('--border_size', type=int, default=15,
                            help='The number of pixels to remove from the borders to compute the repeatability.')
        parser.add_argument('--nms_size', type=int, default=15,
                            help='The NMS size for computing the validation repeatability.')
        parser.add_argument('--num_features', type=int, default=2048,
                            help='The number of desired features to extract.')
        parser.add_argument('--s_mult', type=int, default=60,
                            help='The scale of laf.')
        parser.add_argument('--order_coord', type=str, default='xysr',
                            help='The coordinate order that follows the extracted points. Use yxsr or xysr.')
        parser.add_argument('--heatmap_confidence_threshold', type=float, default=0.001,
                            help='Keypoints confidence threshold.')
        parser.add_argument('--sub_pixel', type=bool, default=True,
                            help='Extract subpixel detection.')
        parser.add_argument('--patch_size', type=int, default=4,
                            help='Subpixel patch size.')
        args = parser.parse_args()
        return args
    # 设置设备和加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_test_config()
    detector =RLFDB(dims=[32, 64, 128, 256], layers=[2, 2, 6, 2], expand_ratio=3.0, mlp_ratio=3.0, use_dw=True,
          drop_path_rate=0.05).to(device)
    checkpoint = torch.load(args.ckpt_file, map_location=torch.device('cpu') if device=='cpu' else None, weights_only=False)
    model_state_disk = checkpoint['model_state']
    detector.load_state_dict(model_state_disk)

    detector = detector.eval()
    # 确保输出根目录存在
    os.makedirs(output_root, exist_ok=True)

    # 遍历所有子文件夹
    for subdir in tqdm(os.listdir(input_root)):
        input_subdir = os.path.join(input_root, subdir)
        output_subdir = os.path.join(output_root, subdir)
        if not os.path.isdir(input_subdir):
            continue
        # 创建对应的输出子目录
        os.makedirs(output_subdir, exist_ok=True)
        # 处理每个.ppm文件
        for filename in os.listdir(input_subdir):
            if filename.endswith('.ppm'):
                input_path = os.path.join(input_subdir, filename)
                output_path = os.path.join(output_subdir, f"{filename}.kpt.npy")
                # 加载图片并检测特征点
                im_rgb = load_im(input_path)
                keypoints = detect_and_save(args, im_rgb, detector, device)
                # 保存特征点
                np.save(output_path, keypoints)

if __name__ == "__main__":
    detect_proposed(   # 设置输入输出路径
    input_root = '/path/to/hpatches-sequences-release',
    output_root = '/results/hpatches-outputs')