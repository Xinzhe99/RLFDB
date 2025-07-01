import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import HSequences_bench.tools.aux_tools as aux
import HSequences_bench.tools.geometry_tools as geo_tools
import HSequences_bench.tools.repeatability_tools as rep_tools
from HSequences_bench.tools.HSequences_reader import HSequences_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='HSequences Compute Repeatability')

    parser.add_argument('--data-dir', type=str, default=r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/hpatches-sequences-release/',
                        help='The root path to HSequences dataset.')

    parser.add_argument('--results-bench-dir', type=str, default='HSequences_bench/results/',
                        help='The output path to save the results.')

    parser.add_argument('--results-dir', type=str, default='/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_feature_match/BALF-V2/compare_diff_methods_on_HSequences/AKAZE_features_hpatch',
                        help='The path to the extracted points.')

    parser.add_argument('--detector-name', type=str, default='DeDoDe',
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

    return parser.parse_args()

def compute_hsequences_metrics(args):
    """
    Computes repeatability metrics for keypoints on the HSequences dataset.

    Args:
        args (argparse.Namespace): An object containing the parsed command-line arguments.

    Returns:
        dict: A dictionary containing the computed repeatability metrics.
              Returns None if no valid results were collected.
    """
    print(f"{args.detector_name}: {args.split}")
    aux.check_directory(args.results_bench_dir)
    data_loader = HSequences_dataset(args.data_dir, args.split, args.split_path)
    results = aux.create_overlapping_results(args.detector_name, args.overlap)
    count_seq = 0

    for sample_id, sample_data in enumerate(data_loader.extract_hsequences()):
        sequence = sample_data['sequence_name']
        count_seq += 1
        image_src = sample_data['im_src']
        images_dst = sample_data['images_dst']
        h_src_2_dst = sample_data['h_src_2_dst']
        h_dst_2_src = sample_data['h_dst_2_src']

        print(f"\nComputing {sequence} sequence {count_seq} / {len(data_loader.sequences)} \n")

        try:
            for idx_im in tqdm(range(len(images_dst))):
                mask_src, mask_dst = geo_tools.create_common_region_masks(h_dst_2_src[idx_im], image_src.shape, images_dst[idx_im].shape)
                src_pts_filename = os.path.join(args.results_dir, f'{sample_data["sequence_name"]}/1.ppm.kpt.npy')
                dst_pts_filename = os.path.join(args.results_dir, f'{sample_data["sequence_name"]}/{idx_im+2}.ppm.kpt.npy')

                if not os.path.isfile(src_pts_filename):
                    print(f"Could not find the file: {src_pts_filename}")
                    continue
                if not os.path.isfile(dst_pts_filename):
                    print(f"Could not find the file: {dst_pts_filename}")
                    continue

                src_pts = np.load(src_pts_filename)
                dst_pts = np.load(dst_pts_filename)

                if args.order_coord == 'xysr':
                    src_pts = np.asarray([[x[1], x[0], x[2], x[3]] for x in src_pts])
                    dst_pts = np.asarray([[x[1], x[0], x[2], x[3]] for x in dst_pts])

                src_idx = rep_tools.check_common_points(src_pts, mask_src)
                src_pts = src_pts[src_idx]
                dst_idx = rep_tools.check_common_points(dst_pts, mask_dst)
                dst_pts = dst_pts[dst_idx]

                if args.top_k_points:
                    src_idx = rep_tools.select_top_k(src_pts, args.top_k_points)
                    src_pts = src_pts[src_idx]
                    dst_idx = rep_tools.select_top_k(dst_pts, args.top_k_points)
                    dst_pts = dst_pts[dst_idx]

                src_pts = np.asarray([[x[1], x[0], x[2], x[3]] for x in src_pts])
                dst_pts = np.asarray([[x[1], x[0], x[2], x[3]] for x in dst_pts])

                src_to_dst_pts = geo_tools.apply_homography_to_points(src_pts, h_src_2_dst[idx_im])
                dst_to_src_pts = geo_tools.apply_homography_to_points(dst_pts, h_dst_2_src[idx_im])

                if args.dst_to_src_evaluation:
                    points_src = src_pts
                    points_dst = dst_to_src_pts
                else:
                    points_src = src_to_dst_pts
                    points_dst = dst_pts

                repeatability_results = rep_tools.compute_repeatability(points_src, points_dst, overlap_err=1-args.overlap,
                                                                        dist_match_thresh=args.pixel_threshold)

                results['rep_single_scale'].append(repeatability_results['rep_single_scale'])
                results['rep_multi_scale'].append(repeatability_results['rep_multi_scale'])
                results['num_points_single_scale'].append(repeatability_results['num_points_single_scale'])
                results['num_points_multi_scale'].append(repeatability_results['num_points_multi_scale'])
                results['error_overlap_single_scale'].append(repeatability_results['error_overlap_single_scale'])
                results['error_overlap_multi_scale'].append(repeatability_results['error_overlap_multi_scale'])

        except Exception as e:
            print(f"Error processing sequence {sequence}: {e}")
            continue

    if results['rep_single_scale']:
        rep_single = np.array(results['rep_single_scale']).mean()
        rep_multi = np.array(results['rep_multi_scale']).mean()
        error_overlap_s = np.array(results['error_overlap_single_scale']).mean()
        error_overlap_m = np.array(results['error_overlap_multi_scale']).mean()
        num_features = np.array(results['num_points_single_scale']).mean()

        final_results = {
            'overlap_threshold': args.overlap,
            'repeatability_multi_scale': rep_multi,
            'repeatability_single_scale': rep_single,
            'overlap_error_multi_scale': error_overlap_s,
            'overlap_error_single_scale': error_overlap_m,
            'average_num_features': num_features,
            'raw_results': results  # Optionally return the raw per-image pair results
        }

        # print('\n## Overlap @{0}:\n \
        #        #### Rep. Multi: {1:.4f}\n \
        #        #### Rep. Single: {2:.4f}\n \
        #        #### Overlap Multi: {3:.4f}\n \
        #        #### Overlap Single: {4:.4f}\n \
        #        #### Num Feats: {5:.4f}'.format(
        #     args.overlap, rep_multi, rep_single, error_overlap_s, error_overlap_m, num_features))

        # output_file_path = os.path.join(args.results_bench_dir, f'{args.detector_name}_{args.split}.pickle')
        # with open(output_file_path, 'wb') as handle:
        #     pickle.dump(final_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return final_results
    else:
        print("\nNo valid results were collected due to missing or problematic npy files.")
        return None

if __name__ == '__main__':
    arguments = parse_arguments()
    results = compute_hsequences_metrics(arguments)
    if results:
        print("\nFinal Results:")
        for key, value in results.items():
            if key != 'raw_results':  # Optionally skip printing raw results
                print(f"  {key}: {value}")