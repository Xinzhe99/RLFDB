import cv2
import numpy as np
import random
import torch
from imgaug import augmenters as iaa

from benchmark_test import repeatability_tools, geometry_tools


perms = ((0, 1, 2), (0, 2, 1),
         (1, 0, 2), (1, 2, 0),
         (2, 0, 1), (2, 1, 0))


def ratio_preserving_resize(img, target_size):
    scales = np.array((target_size[0]/img.shape[0], target_size[1]/img.shape[1]))##h_s,w_s

    new_size = np.round(np.array(img.shape[:2])*np.max(scales)).astype(np.int)#
    temp_img = cv2.resize(img, tuple(new_size[::-1]))
    curr_h, curr_w = temp_img.shape[:2]
    target_h, target_w = target_size
    ##
    hp = (target_h-curr_h)//2
    wp = (target_w-curr_w)//2
    aug = iaa.Sequential([iaa.CropAndPad(px=(hp, wp, target_h-curr_h-hp, target_w-curr_w-wp),keep_size=False)])
    new_img = aug(image=temp_img)
    return new_img


def adapt_homography_to_preprocessing(zip_data, args):
    '''缩放后对应的图像的homography矩阵
    :param zip_data:{'shape':原图像HW,
                        'warped_shape':warped图像HW,
                        'homography':原始变换矩阵}
    :return:对应当前图像尺寸的homography矩阵
    '''
    H = zip_data['homography'].astype(np.float32)
    source_size = zip_data['shape'].astype(np.float32)#h,w
    source_warped_size = zip_data['warped_shape'].astype(np.float32)#h,w
    target_size = np.array(args.resize_shape,dtype=np.float32)#h,w

    # Compute the scaling ratio due to the resizing for both images
    s = np.max(target_size/source_size)
    up_scale = np.diag([1./s, 1./s, 1])
    warped_s = np.max(target_size/source_warped_size)
    down_scale = np.diag([warped_s, warped_s, 1])

    # Compute the translation due to the crop for both images
    pad_y, pad_x = (source_size*s - target_size)//2.0

    translation = np.array([[1, 0, pad_x],
                            [0, 1, pad_y],
                            [0, 0, 1]],dtype=np.float32)
    pad_y, pad_x = (source_warped_size*warped_s - target_size) //2.0

    warped_translation = np.array([[1,0, -pad_x],
                                    [0,1, -pad_y],
                                    [0,0,1]], dtype=np.float32)
    H = warped_translation @ down_scale @ H @ up_scale @ translation
    return H



def read_bgr_image(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return img_bgr.reshape(img_bgr.shape[0], img_bgr.shape[1], 3)

def bgr_to_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.reshape(img_rgb.shape[0], img_rgb.shape[1], 3)

def rgb_to_bgr(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr.reshape(img_bgr.shape[0], img_bgr.shape[1], 3)

def bgr_photometric_to_rgb(im_bgr):
    img_distorsion = bgr_distorsion(im_bgr)
    img_rgb = cv2.cvtColor(img_distorsion, cv2.COLOR_BGR2RGB)
    return img_rgb.reshape(img_rgb.shape[0], img_rgb.shape[1], 3)
def rgb_photometric_distortion(im_rgb):
    """
    对 RGB 图像应用 bgr_distorsion 中定义的光度畸变。
    先将 RGB 转换为 BGR，应用畸变，再转换回 RGB。
    """
    img_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    img_distorted_bgr = bgr_distorsion(img_bgr)
    img_distorted_rgb = cv2.cvtColor(img_distorted_bgr, cv2.COLOR_BGR2RGB)
    return img_distorted_rgb.reshape(img_distorted_rgb.shape[0], img_distorted_rgb.shape[1], 3)

def bgr_distorsion(image, lower=0.5, upper=1.5, delta=18.0, delta_brigtness=36):
    
    image = image.astype(float)

    if np.random.randint(2):
        delta = np.random.uniform(-delta_brigtness, delta_brigtness)
        image += delta
        image = check_margins(image)

    contrast = np.random.randint(2)
    if contrast:
        alpha = np.random.uniform(lower, upper)
        image *= alpha
        image = check_margins(image)

    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image.astype(float)

    if np.random.randint(2):
        image[:, :, 1] *= np.random.uniform(lower, upper)
        image = check_margins(image, axis=1)
    if np.random.randint(2):
        image[:, :, 0] += np.random.uniform(-delta, delta)
        image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = image.astype(float)

    if contrast:
        alpha = np.random.uniform(lower, upper)
        image *= alpha
        image = check_margins(image)

    if np.random.randint(2):
        swap = perms[np.random.randint(len(perms))]
        image = swap_channels(image, swap)  # shuffle channels

    return image.astype(np.uint8)


def rotate_image_and_points(image, points, angle):
    """
    旋转图像和关键点
    Args:
        image: 输入图像 shape=(H, W, C)
        points: 关键点坐标 shape=(N, 4) 其中前两列是xy坐标，后两列是scale和response
        angle: 旋转角度
    Returns:
        rotated_image: 旋转后的图像
        rotated_points: 旋转后的关键点坐标
    """
    # 获取图像中心
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋转图像
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    if points is None:
        return rotated_image, None

    # 只取前两列坐标进行旋转变换
    xy_points = points[:, :2]  # 取出坐标部分
    other_info = points[:, 2:]  # 保存scale和response信息

    # 转换点坐标为齐次坐标
    ones = np.ones(shape=(len(xy_points), 1))
    points_ones = np.hstack([xy_points, ones])  # shape=(N, 3)

    # 旋转点坐标
    rotated_xy = rotation_matrix.dot(points_ones.T).T  # shape=(N, 2)

    # 裁剪超出边界的点
    valid_points = (rotated_xy[:, 0] >= 0) & (rotated_xy[:, 0] < width) & \
                   (rotated_xy[:, 1] >= 0) & (rotated_xy[:, 1] < height)

    # 将旋转后的坐标与scale和response信息重新组合
    rotated_points = np.hstack([rotated_xy[valid_points], other_info[valid_points]])

    return rotated_image, rotated_points



def add_noise(image, p=0.1):
    # 添加概率控制
    if np.random.random() > p:  # 1-p的概率不添加噪声
        return image

    # 随机生成0-1之间的scale值
    scale = np.random.uniform(0, 1)

    # 有p的概率添加高斯噪声
    noise = np.random.normal(0, scale, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    return image

def add_shadow(image, p=0.1):
    # 添加概率控制
    if np.random.random() > p:
        return image

    image = image.copy()
    height, width = image.shape[:2]

    # 随机生成阴影多边形的顶点数（3-5个顶点）
    num_vertices = np.random.randint(3, 7)

    # 随机生成多边形顶点
    vertices = []
    for _ in range(num_vertices):
        vertex = (np.random.randint(0, width), np.random.randint(0, height))
        vertices.append(vertex)
    vertices = np.array([vertices], dtype=np.int32)

    # 创建阴影遮罩
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillPoly(mask, vertices, 255)

    # 随机阴影强度（0.3-0.7之间的随机值）
    intensity = np.random.uniform(0.3, 0.7)

    # 应用阴影
    for c in range(3):  # 对每个颜色通道应用阴影
        image[:, :, c][mask == 255] = image[:, :, c][mask == 255] * intensity

    return image.astype(np.uint8)
def check_margins(img, axis=-1):
    if axis == -1:
        img[img > 255.0] = 255.0
        img[img < 0.0] = 0.0
    else:
        img[:, :, axis][img[:, :, axis] > 255.0] = 255.0
        img[:, :, axis][img[:, :, axis] < 0.0] = 0.0
    return img

def swap_channels(image, swaps):
    image = image[:, :, swaps]
    return image


def generate_homography(IMAGE_SHAPE, hom_config):

    src_point = np.array([[               0,                0],
                          [IMAGE_SHAPE[1]-1,                0],
                          [               0, IMAGE_SHAPE[0]-1],
                          [IMAGE_SHAPE[1]-1, IMAGE_SHAPE[0]-1]], dtype=np.float32)

    dst_point = get_dst_point(hom_config['perspective'], IMAGE_SHAPE)

    rotation = hom_config['rotation']
    rot = random.randint(-rotation, rotation)

    scale = 1.0 + hom_config['scale'] * random.randint(-25, 50) * 0.1

    center_offset = 40
    center = (IMAGE_SHAPE[1] / 2 + random.randint(-center_offset, center_offset),
              IMAGE_SHAPE[0] / 2 + random.randint(-center_offset, center_offset))

    RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
    f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
    homography = cv2.getPerspectiveTransform(src_point, f_point)
    
    return homography

def get_dst_point(perspective, IMAGE_SHAPE):
    a = random.random()
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()

    if random.random() > 0.5:
        left_top_x = perspective*a
        left_top_y = perspective*b
        right_top_x = 0.9+perspective*c
        right_top_y = perspective*d
        left_bottom_x  = perspective*a
        left_bottom_y  = 0.9 + perspective*e
        right_bottom_x = 0.9 + perspective*c
        right_bottom_y = 0.9 + perspective*f
    else:
        left_top_x = perspective*a
        left_top_y = perspective*b
        right_top_x = 0.9+perspective*c
        right_top_y = perspective*d
        left_bottom_x  = perspective*e
        left_bottom_y  = 0.9 + perspective*b
        right_bottom_x = 0.9 + perspective*f
        right_bottom_y = 0.9 + perspective*d

    dst_point = np.array([(IMAGE_SHAPE[1]*left_top_x,IMAGE_SHAPE[0]*left_top_y,1),
            (IMAGE_SHAPE[1]*right_top_x, IMAGE_SHAPE[0]*right_top_y,1),
            (IMAGE_SHAPE[1]*left_bottom_x,IMAGE_SHAPE[0]*left_bottom_y,1),
            (IMAGE_SHAPE[1]*right_bottom_x,IMAGE_SHAPE[0]*right_bottom_y,1)],dtype = 'float32')
    return dst_point

def apply_homography_to_source_image(source_im, h):
    shape_source_im = source_im.shape
    dst = cv2.warpPerspective(source_im, h, (shape_source_im[1], shape_source_im[0]))
    return dst

def apply_homography_to_source_labels_torch(points, IMAGE_SHAPE, homography, bilinear = False):
    """将关键点通过单应性变换映射到目标图像空间并生成热图
    Args:
        points: 原始关键点坐标 shape=(N,2)
        IMAGE_SHAPE: 图像尺寸 (H,W)
        homography: 单应性变换矩阵 shape=(3,3)
        bilinear: 是否使用双线性插值，默认False
    Returns:
        warped_heatmap: 变换后的热图 shape=(1,H,W)
    """
    # 获取图像高宽
    H, W = IMAGE_SHAPE[0], IMAGE_SHAPE[1]
    
    # 将输入点转换为长整型tensor
    if isinstance(points, torch.Tensor):
        points = points.long() 
    else:
        points = torch.tensor(points).long()
        
    # 将点坐标堆叠并进行单应性变换
    warped_pnts = warp_points(torch.stack((points[:, 0], points[:, 1]), dim=1), homography)
    
    outs = {}
    # 如果使用双线性插值
    if bilinear == True:
        # 生成双线性插值后的标签
        warped_labels_bi = get_labels_bi(warped_pnts, H, W)
        outs['labels_bi'] = warped_labels_bi

    # 过滤掉超出图像范围的点
    warped_pnts = filter_points(warped_pnts, torch.tensor([W, H]))
    
    # 生成热图：将变换后的点映射到对应位置并设为1
    warped_heatmap = scatter_points(warped_pnts, H, W, res_ext = 1)
    
    return warped_heatmap

def warp_points(points, homographies, device='cpu'):
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    # homographies = homographies.unsqueeze(0) if len(homographies.shape) == 2 else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)
    # warped_points = homographies*points
    # points = points.double()
    warped_points = homographies@points.transpose(0,1)
    # warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

def get_labels_bi(warped_pnts, H, W):
    pnts_ext, res_ext = extrapolate_points(warped_pnts)
    pnts_ext, mask = filter_points(pnts_ext, torch.tensor([W, H]), return_mask=True)
    res_ext = res_ext[mask]
    warped_labels_bi = scatter_points(pnts_ext, H, W, res_ext = res_ext)
    return warped_labels_bi

def extrapolate_points(pnts):
    pnts_int = pnts.long().type(torch.FloatTensor)
    pnts_x, pnts_y = pnts_int[:,0], pnts_int[:,1]

    stack_1 = lambda x, y: torch.stack((x, y), dim=1)
    pnts_ext = torch.cat((pnts_int, stack_1(pnts_x, pnts_y+1),
        stack_1(pnts_x+1, pnts_y), pnts_int+1), dim=0)

    pnts_res = pnts - pnts_int # (x, y)
    x_res, y_res = pnts_res[:,0], pnts_res[:,1] # residuals
    res_ext = torch.cat(((1-x_res)*(1-y_res), (1-x_res)*y_res, 
            x_res*(1-y_res), x_res*y_res), dim=0)
    return pnts_ext, res_ext

def filter_points(points, shape, return_mask=False):
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points [mask]

def scatter_points(warped_pnts, H, W, res_ext = 1):
    quan = lambda x: x.round().long() 
    warped_labels = torch.zeros(H, W)
    warped_labels[quan(warped_pnts)[:, 1], quan(warped_pnts)[:, 0]] = res_ext
    warped_labels = warped_labels.view(-1, H, W)
    return warped_labels

def select_k_best(points, k):
    """ Select the k most probable points (and strip their proba).
    points has shape (num_points, 3) where the last coordinate is the proba. """
    sorted_prob = points
    if points.shape[1] > 2 and k != 0:
        # sorted_prob = points[points[:, 2].argsort(), :2]
        sorted_prob = points[points[:, 2].argsort(), :]
        start = min(k, points.shape[0])
        sorted_prob = sorted_prob[-start:, :]
    return sorted_prob

def labels_to_heatmap(points, IMAGE_SHAPE):
    heatmap = np.zeros((IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    points = points.astype(int)
    heatmap[points[:, 1], points[:, 0]] = 1
    return heatmap.astype('float32')


def get_window_point(shape, patch_size, crop_type='random'):
    h, w, _ = shape
    # 确保patch_size是整数
    half_size = patch_size // 2

    # 计算中心点的有效范围
    min_h = half_size
    max_h = h - half_size
    min_w = half_size
    max_w = w - half_size

    if crop_type == 'random':
        # 确保随机点在有效范围内
        window_h = random.randint(min_h, max_h)
        window_w = random.randint(min_w, max_w)
    elif crop_type=='center':
        window_h = h // 2
        window_w = w // 2

    return np.array([window_h, window_w])
# def get_window_point(shape, patch_size, crop_type='random'):
#     h, w, _ = shape
#     if crop_type == 'random':
#         window_h = random.randint(patch_size/2, h - patch_size/2)
#         window_w = random.randint(patch_size/2, w - patch_size/2)
#     else:
#         window_h = h / 2
#         window_w = w / 2
#
#     return np.array([window_h, window_w])

def debug_synthetic_pairs_repeatability(heatmap_src, heatmap_dst, image_src, image_dst, h_dst_2_src):
    mask_src, mask_dst = geometry_tools.create_common_region_masks(h_dst_2_src, image_src.shape, image_dst.shape)

    src_pts = heatmap_to_points(heatmap_src)
    dst_pts = heatmap_to_points(heatmap_dst)

    src_pts = np.asarray(list(map(lambda x: [x[1], x[0], 1.0, 1.0], src_pts)))
    dst_pts = np.asarray(list(map(lambda x: [x[1], x[0], 1.0, 1.0], dst_pts)))

    src_idx = repeatability_tools.check_common_points(src_pts, mask_src)
    src_pts = src_pts[src_idx]
    # print('src_idx: ', src_idx)

    dst_idx = repeatability_tools.check_common_points(dst_pts, mask_dst)
    dst_pts = dst_pts[dst_idx]
    # print('src_idx: ', src_idx)

    src_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], src_pts)))
    dst_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], dst_pts)))


    dst_to_src_pts = geometry_tools.apply_homography_to_points(
        dst_pts, h_dst_2_src)

    
    repeatability_results = repeatability_tools.compute_repeatability(src_pts, dst_to_src_pts, overlap_err=1-0.6,
                                                                dist_match_thresh=5)

    print("\n rep_s {:.2f} , rep_m {:.2f}, p_s {:d} , p_m {:d}, eps_s {:.2f}, eps_m {:.2f}, p_all {:d}"
        .format(
            repeatability_results['rep_single_scale'], repeatability_results['rep_multi_scale'], repeatability_results['num_points_single_scale'], 
            repeatability_results['num_points_multi_scale'], repeatability_results['error_overlap_single_scale'], repeatability_results['error_overlap_multi_scale'],
            repeatability_results['total_num_points']
    ))

def heatmap_to_points(heatmap):
    ys, xs = np.where(heatmap == 1.0)
    if len(xs) == 0:
        return np.zeros((0, 2))
    pts = np.zeros((len(xs), 2))
    pts[:, 0] = xs
    pts[:, 1] = ys

    return pts # [N,2]


def make_shape_even(image):
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)
    return image

def mod_padding_symmetric(image, factor=64):
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = ((height + factor) // factor) * factor, (
        (width + factor) // factor) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = np.pad(
        image, ((padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)),
        mode='constant', constant_values=0)
    return image


def debug_test_results(image_RGB_norm, image_even, image_pad, score_map_pad_np, score_map, score_map_remove_border, score_map_nms):
    print('original image shape: ', image_RGB_norm.shape[0], image_RGB_norm.shape[1])
    print('even image shape: ', image_even.shape[0], image_even.shape[1])
    print('pad image shape: ', image_pad.shape[0], image_pad.shape[1])
    print('pad score map shape', score_map_pad_np.shape)
    print('score map shape', score_map.shape)
    print('remove border score map shape', score_map_remove_border.shape)
    print('nms score map shape', score_map_nms.shape)

    assert(image_RGB_norm.shape[0] == score_map.shape[0] and image_RGB_norm.shape[1] == score_map.shape[1])
    assert(image_pad.shape[0] == score_map_pad_np.shape[0] and image_pad.shape[1] == score_map_pad_np.shape[1])
    assert(image_RGB_norm.shape[0] == score_map_remove_border.shape[0] and image_RGB_norm.shape[1] == score_map_remove_border.shape[1])
    assert(image_RGB_norm.shape[0] == score_map_nms.shape[0] and image_RGB_norm.shape[1] == score_map_nms.shape[1])

    import matplotlib.pyplot as plt
    fig = plt.figure()
    rows = 2; cols = 4
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(image_RGB_norm)
    ax1.set_title('image_RGB')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(image_even)
    ax2.set_title('image_even')
    ax2.axis("off")

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(image_pad)
    ax3.set_title('image_pad')
    ax3.axis("off")

    ax4 = fig.add_subplot(rows, cols, 4)
    ax4.imshow(score_map_pad_np, cmap='gray')
    ax4.set_title('score_map_pad')
    ax4.axis("off")

    ax5 = fig.add_subplot(rows, cols, 5)
    ax5.imshow(score_map, cmap='gray')
    ax5.set_title('score_map')
    ax5.axis("off")

    ax6 = fig.add_subplot(rows, cols, 6)
    ax6.imshow(score_map_remove_border, cmap='gray')
    ax6.set_title('remove_border')
    ax6.axis("off")

    ax7 = fig.add_subplot(rows, cols, 7)
    ax7.imshow(score_map_nms, cmap='gray')
    ax7.set_title('nms')
    ax7.axis("off")

    plt.show()


def debug_test_multiscale_results(image_RGB_norm, image_even, image_pad, score_map_pad_np, score_map):
    print('original image shape: ', image_RGB_norm.shape[0], image_RGB_norm.shape[1])
    print('even image shape: ', image_even.shape[0], image_even.shape[1])
    print('pad image shape: ', image_pad.shape[0], image_pad.shape[1])
    print('pad score map shape', score_map_pad_np.shape)
    print('score map shape', score_map.shape)


    assert(image_RGB_norm.shape[0] == score_map.shape[0] and image_RGB_norm.shape[1] == score_map.shape[1])
    assert(image_pad.shape[0] == score_map_pad_np.shape[0] and image_pad.shape[1] == score_map_pad_np.shape[1])

    import matplotlib.pyplot as plt
    fig = plt.figure()
    rows = 2; cols = 3
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(image_RGB_norm)
    ax1.set_title('image_RGB')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(image_even)
    ax2.set_title('image_even')
    ax2.axis("off")

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(image_pad)
    ax3.set_title('image_pad')
    ax3.axis("off")

    ax4 = fig.add_subplot(rows, cols, 4)
    ax4.imshow(score_map_pad_np, cmap='gray')
    ax4.set_title('score_map_pad')
    ax4.axis("off")

    ax5 = fig.add_subplot(rows, cols, 5)
    ax5.imshow(score_map, cmap='gray')
    ax5.set_title('score_map')
    ax5.axis("off")

    plt.show()