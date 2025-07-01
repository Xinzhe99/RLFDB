import argparse
from utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='General blur feature detection')

    # Basic configuration
    parser.add_argument('--cfg_file', type=str, default=r'configs/gopro_train_detection.yaml')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Path for saving checkpoints')
    parser.add_argument('--exper_name', type=str,default='train_final_reinforment')
    parser.add_argument('--fix_random_seed', type=bool, default=True,
                        help='The random seed value for PyTorch and Numpy.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='The random seed value for PyTorch and Numpy.')
    parser.add_argument('--is_debugging', type=bool, default=False,
                        help='Set variable to True if you desire to train network on a smaller dataset.')
    parser.add_argument('--resume_training', type=str, default=None,
                        help='Set saved model parameters if resume training is desired.')
    parser.add_argument('--val_data_dir', type=str, default='data/hpatches-sequences-release/',
                        help='The root path to HSequences dataset.')
    parser.add_argument('--split', type=str, default='full',
                        help='The name of the HPatches (HSequences) split. Use full, debug_view, debug_illum, view or illum.')
    parser.add_argument('--split_path', type=str, default='benchmark_test/splits.json',
                        help='The path to the split json file.')
    # 添加GPU选择参数
    parser.add_argument('--gpu_ids', type=str, default='1',
                        help='GPU IDs to use. 格式为: "0"使用cuda:0, "1"使用cuda:1, "0,1"同时使用两个GPU. 不填则使用CPU.')

    #训练的时候顺便测评了,以下是测评相关的参数
    parser.add_argument('--benchmark_input_root', type=str, default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/hpatches-sequences-release',
                        help='Path for saving checkpoints')
    parser.add_argument('--border_size', type=int, default=15,
                        help='The number of pixels to remove from the borders to compute the repeatability.')
    parser.add_argument('--nms_size', type=int, default=15,
                        help='The NMS size for computing the validation repeatability.')
    parser.add_argument('--sub_pixel', type=bool, default=True,
                        help='Extract subpixel detection.')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Subpixel patch size.')
    parser.add_argument('--order_coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use yxsr or xysr.')
    parser.add_argument('--heatmap_confidence_threshold', type=float, default=0.001,
                        help='Keypoints confidence threshold.')
    parser.add_argument('--num_features', type=int, default=2048,
                        help='The number of desired features to extract.')
    args = parser.parse_args()

    cfg = common_utils.get_cfg_from_yaml_file(args.cfg_file)

    return args, cfg


def parse_test_config():
    parser = argparse.ArgumentParser(description='motion blur feature matching test')

    # Basic configuration
    parser.add_argument('--cfg_file', type=str, default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_feature_match/BALF/balf/configs/test.yaml')
    parser.add_argument('--ckpt_file', type=str, default = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_feature_match/BALF-V2/pretrained/balf/balf.pth',
                        help='The path to the checkpoint file to load the detector weights.')
    parser.add_argument('--ckpt_descriptor_file', type=str, default = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_feature_match/BALF/balf/pretrained/hardnet/HardNet++.pth',
                        help='The path to the checkpoint file to load the descriptor weights.')
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

    cfg = common_utils.get_cfg_from_yaml_file(args.cfg_file)

    return args, cfg

def parse_hsequences_metrics():
    parser = argparse.ArgumentParser(description='HSequences Compute Repeatability')

    parser.add_argument('--data-dir', type=str, default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/hpatches-sequences-release/',
                        help='The root path to HSequences dataset.')

    parser.add_argument('--results-bench-dir', type=str, default='HSequences_bench/results/',
                        help='The output path to save the results.')

    parser.add_argument('--results-dir', type=str, default='/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_feature_match/BALF-V2/add_expert_random_blur_eval_hpatch_kernel_7_epoch_23',
                        help='The path to the extracted points.')

    parser.add_argument('--detector-name', type=str, default='ori_balf',
                        help='The name of the detector to compute metrics.')

    parser.add_argument('--split', type=str, default='full',
                        help='The name of the HPatches (HSequences) split. Use full, debug_view, debug_illum, view or illum.')

    parser.add_argument('--split-path', type=str, default='HSequences_bench/splits.json',
                        help='The path to the split json file.')

    parser.add_argument('--top-k-points', type=int, default=1000,
                        help='The number of top points to use for evaluation. Set to None to use all points')

    parser.add_argument('--overlap', type=float, default=0.6,
                        help='The overlap threshold for a correspondence to be considered correct.')

    parser.add_argument('--pixel-threshold', type=int, default=5,
                        help='The distance of pixels for a matching correspondence to be considered correct.')

    parser.add_argument('--dst-to-src-evaluation', type=bool, default=True,
                        help='Order to apply homography to points. Use True for dst to src, False otherwise.')

    parser.add_argument('--order-coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use either xysr or yxsr.')

    args = parser.parse_args()
    return args