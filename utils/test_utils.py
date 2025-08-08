import numpy as np
import torch
import yaml
from scipy.ndimage.filters import maximum_filter

def detect(args, im, detector, device):
    """Optimized version of detect_and_save function with reduced variable assignments and efficient operations"""
    # Normalize image and get dimensions in one step
    im_norm = im / 255.0
    height_orig, width_orig = im_norm.shape[:2]
    
    # Chain preprocessing operations
    image_processed = mod_padding_symmetric(make_shape_even(im_norm), factor=64)
    
    # Direct tensor conversion with efficient operations
    input_tensor = torch.from_numpy(image_processed).permute(2, 0, 1).unsqueeze(0).float()
    
    # Model inference
    with torch.inference_mode():
        score_map_padded = detector(input_tensor.to(device))['prob'][0].detach().cpu().numpy()
    
    # Efficient unpadding using calculated offsets
    pad_h, pad_w = score_map_padded.shape[:2]
    h_offset = (pad_h - height_orig) >> 1  # Use bit shift for division by 2
    w_offset = (pad_w - width_orig) >> 1
    
    # Extract original region and remove borders in one step
    score_map_cropped = remove_borders(
        score_map_padded[h_offset:h_offset + height_orig, w_offset:w_offset + width_orig],
        borders=args.border_size
    )
    
    # Extract and sort points efficiently
    pts = get_points_direct_from_score_map(
        heatmap=score_map_cropped,
        conf_thresh=args.heatmap_confidence_threshold,
        nms_size=args.nms_size,
        subpixel=args.sub_pixel,
        patch_size=args.patch_size,
        order_coord=args.order_coord
    )
    
    # Early return for empty results
    if pts.size == 0:
        return np.zeros([0, 4])
    
    # Sort and limit points in one operation
    return pts[np.argsort(-pts[:, 3])][:args.num_features]


def get_cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)

    return config

def make_shape_even(image):
    """Optimized version: only pad if necessary and use bitwise operations for even check"""
    height, width = image.shape[:2]
    
    # Use bitwise AND for faster even/odd check
    padh = height & 1  # equivalent to height % 2
    padw = width & 1   # equivalent to width % 2
    
    # Only pad if necessary
    if padh or padw:
        image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)
    
    return image

def mod_padding_symmetric(image, factor=64):
    """Optimized version: simplified calculations and early return if no padding needed"""
    height, width = image.shape[:2]
    
    # Calculate remainder more efficiently
    height_rem = height % factor
    width_rem = width % factor
    
    # Early return if no padding needed
    if height_rem == 0 and width_rem == 0:
        return image
    
    # Calculate padding needed
    padh = (factor - height_rem) % factor
    padw = (factor - width_rem) % factor
    
    # Only pad if necessary
    if padh or padw:
        # Use integer division for symmetric padding
        pad_top = padh >> 1  # equivalent to padh // 2
        pad_bottom = padh - pad_top
        pad_left = padw >> 1  # equivalent to padw // 2
        pad_right = padw - pad_left
        
        image = np.pad(
            image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant', constant_values=0)
    
    return image

def remove_borders(image, borders): 
    ## Input : [B, H, W, C] or [H, W, C] or [H, W]

    shape = image.shape
    new_im = np.zeros_like(image)
    # if len(shape) == 4:
    #     shape = [shape[1], shape[2], shape[3]]
    #     new_im[:, borders:shape[0]-borders, borders:shape[1]-borders, :] = image[:, borders:shape[0]-borders, borders:shape[1]-borders, :]
    # elif len(shape) == 3:
    if len(shape) == 3:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders, :] = image[borders:shape[0] - borders, borders:shape[1] - borders, :]
    else:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders] = image[borders:shape[0] - borders,  borders:shape[1] - borders]
    return new_im


def apply_nms(score_map, size):
    
    score_map = score_map * (score_map == maximum_filter(score_map, footprint=np.ones((size, size))))

    return score_map

def get_point_coordinates(map, scale_value=1., num_points=1000, threshold=-1, order_coord='xysr'):
    ## input numpy array score map : [H, W]
    indexes = find_index_higher_scores(map, num_points=num_points, threshold=threshold)
    new_indexes = []
    for ind in indexes:

        scores = map[ind[0], ind[1]]
        if order_coord == 'xysr':
            tmp = [ind[1], ind[0], scale_value, scores]
        elif order_coord == 'yxsr':
            tmp = [ind[0], ind[1], scale_value, scores]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes)

def find_index_higher_scores(map, num_points = 1000, threshold = -1):
    # Best n points
    if threshold == -1:

        flatten = map.flatten()
        order_array = np.sort(flatten)

        order_array = np.flip(order_array, axis=0)

        threshold = order_array[num_points-1]
        if threshold <= 0.0:
            indexes = np.argwhere(order_array > 0.0)
            if len(indexes) == 0:
                threshold = 0.0
            else:
                threshold = order_array[indexes[len(indexes)-1]]
        # elif threshold == 0.0:
        #     threshold = order_array[np.nonzero(order_array)].min()

    indexes = np.argwhere(map >= threshold)

    return indexes[:num_points]

def get_points_direct_from_score_map(
    heatmap, conf_thresh=0.015, nms_size=15,
    subpixel=True, patch_size=5, scale_value=1., order_coord='xysr'
):

    H, W = heatmap.shape[0], heatmap.shape[1]
    ys, xs = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((0, 4))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = xs
    pts[1, :] = ys
    pts[2, :] = heatmap[ys, xs]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_size)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.

    if subpixel:
        pts = soft_argmax_points(pts, heatmap, patch_size=patch_size)

    new_indexes = []
    for idx in range(pts.shape[1]):
        if order_coord == 'xysr':
            tmp = [pts[0,idx], pts[1,idx], scale_value, pts[2,idx]]
        elif order_coord == 'yxsr':
            tmp = [pts[1,idx], pts[0,idx], scale_value, pts[2,idx]]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes) # N,4

def nms_fast(in_corners, H, W, dist_thresh):
    # Fast NMS via grid aggregation + windowed local-maximum filtering.
    # - Deduplicate multiple points falling on the same pixel by keeping the highest score
    # - Use maximum_filter with a (2*dist_thresh+1) square window to keep local maxima
    # - Break ties deterministically using tiny coordinate-dependent epsilon (no numba)
    
    # Handle empty input early
    if in_corners.size == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)

    # Round coordinates to integer pixel locations
    xs = np.rint(in_corners[0]).astype(int)
    ys = np.rint(in_corners[1]).astype(int)
    scores = in_corners[2]

    # Keep only points within image bounds
    valid_mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not np.any(valid_mask):
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)

    xs = xs[valid_mask]
    ys = ys[valid_mask]
    scores = scores[valid_mask]
    original_indices = np.nonzero(valid_mask)[0]

    # If only one valid corner, return directly
    if xs.size == 1:
        out = np.vstack((xs, ys, scores)).reshape(3, 1)
        return out, original_indices

    # Deduplicate: keep only the highest-score point per pixel
    linear_idx = ys * W + xs
    order = np.lexsort((scores, linear_idx))  # primary: linear_idx, secondary: scores
    lin_sorted = linear_idx[order]
    cnts = np.bincount(lin_sorted - lin_sorted.min())  # not directly useful for groups; use unique below
    uniq_lin, group_starts, group_counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    # Index of max-score per pixel is the last in each group (due to secondary sort by score)
    best_in_group_order_idx = group_starts + group_counts - 1
    selected_order = order[best_in_group_order_idx]

    sel_xs = xs[selected_order]
    sel_ys = ys[selected_order]
    sel_scores = scores[selected_order]
    sel_inds = original_indices[selected_order]

    # Build score grid and index grid for selected unique pixels
    score_grid = np.zeros((H, W), dtype=np.float64)
    score_grid[sel_ys, sel_xs] = sel_scores

    index_grid = np.full((H, W), -1, dtype=int)
    index_grid[sel_ys, sel_xs] = sel_inds

    # Add tiny coordinate-dependent epsilon to break ties deterministically
    if sel_xs.size > 0:
        eps = 1e-7
        ranks = (sel_ys.astype(np.int64) * W + sel_xs.astype(np.int64)).astype(np.float64)
        if ranks.size > 1:
            rmin = ranks.min()
            rmax = ranks.max()
            denom = (rmax - rmin) if (rmax > rmin) else 1.0
            ranks = (ranks - rmin) / denom
        else:
            ranks = ranks * 0.0
        score_grid_with_eps = score_grid.copy()
        score_grid_with_eps[sel_ys, sel_xs] += eps * ranks
    else:
        score_grid_with_eps = score_grid

    # Windowed local-maximum filtering
    k = int(dist_thresh) * 2 + 1
    if k < 1:
        k = 1
    local_max = maximum_filter(score_grid_with_eps, size=(k, k), mode='constant', cval=0.0)
    keep_mask = (score_grid_with_eps == local_max) & (score_grid > 0)

    keep_y, keep_x = np.where(keep_mask)
    if keep_x.size == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)

    keep_scores = score_grid[keep_y, keep_x]
    keep_indices = index_grid[keep_y, keep_x]

    # Sort by score descending to match previous behavior
    sort_desc = np.argsort(-keep_scores)
    keep_x = keep_x[sort_desc]
    keep_y = keep_y[sort_desc]
    keep_scores = keep_scores[sort_desc]
    keep_indices = keep_indices[sort_desc]

    out = np.vstack((keep_x, keep_y, keep_scores))
    out_inds = keep_indices
    return out, out_inds

def soft_argmax_points(pts, heatmap, patch_size=5):
    # 减少不必要的拷贝，直接转置
    pts_transposed = pts.transpose()
    patches = extract_patch_from_points(heatmap, pts_transposed, patch_size=patch_size)
    
    # 直接创建tensor，避免先stack再转换
    patches_torch = torch.tensor(np.stack(patches), dtype=torch.float32).unsqueeze(0)
    
    # 合并norm和log操作以减少中间变量
    patches_torch = norm_patches(patches_torch)
    patches_torch = do_log(patches_torch)
    
    # 计算亚像素偏移
    dxdy = soft_argmax_2d(patches_torch, normalized_coordinates=False)
    
    # 直接在原始数据上操作，避免额外拷贝
    offset = patch_size // 2
    dxdy_np = dxdy.numpy().squeeze()
    
    # 创建结果点，避免多次拷贝
    pts_subpixel = pts_transposed.copy()
    pts_subpixel[:, :2] += dxdy_np - offset
    
    return pts_subpixel.transpose()

def extract_patch_from_points(heatmap, points, patch_size=5):
    if type(heatmap) is torch.Tensor:
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = heatmap.squeeze()  # [H, W]
    pad_size = int(patch_size/2)
    heatmap = np.pad(heatmap, pad_size, 'constant')
    patches = []
    ext = lambda img, pnt, wid: img[pnt[1]:pnt[1]+wid, pnt[0]:pnt[0]+wid]
    for i in range(points.shape[0]):
        patch = ext(heatmap, points[i,:].astype(int), patch_size)
        patches.append(patch)

    return patches

def soft_argmax_2d(patches, normalized_coordinates=True):
    import torchgeometry as tgm
    m = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)
    coords = m(patches)  # 1x4x2
    return coords

def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = patches.view(-1, 1, patch_size*patch_size)
    d = torch.sum(patches, dim=-1).unsqueeze(-1) + 1e-6
    patches = patches/d
    patches = patches.view(-1, 1, patch_size, patch_size)
    return patches

def do_log(patches):
    patches[patches<0] = 1e-6
    patches_log = torch.log(patches)
    return patches_log