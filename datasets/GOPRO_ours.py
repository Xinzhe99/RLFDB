import os
import logging
import numpy as np
from pathlib import Path
from datasets import COCO
from datasets.blur_augmentation import BlurAugmentation
import torch
from torchvision.transforms import ToTensor, ToPILImage
import os
import numpy as np
from pathlib import Path
import torch
from datasets import dataset_utils
from datasets import base_dataset


class GOPRO_ours(base_dataset.base_dataset):
    def get_sequence_names(self):
        """获取数据集中的有效序列名称
        根据任务类型(训练/验证)加载对应的序列列表
        """
        # 根据任务类型选择序列文件
        sequences_file = (
            Path(self.dataset_cfg['sequences_split_path'], 'train.txt')
            if self.task == 'train'
            else Path(self.dataset_cfg['sequences_split_path'], 'val.txt')
        )

        # 读取序列名称
        self.sequences_name = open(sequences_file).read()
        logging.info("Get {} sequences from {}".format(
            len(open(sequences_file).readlines()), sequences_file))

    def get_image_paths(self):
        """获取数据集中的有效图像路径
        遍历数据集目录，收集所有符合条件的图像路径
        Returns:
            np.ndarray: 随机打乱后的图像路径数组
        """
        images_info = []

        # 遍历数据集目录
        logging.info('Get {} images from the below path:'.format(self.task))
        for r, d, f in os.walk(self.dataset_cfg['images_path']):
            # 根据任务类型处理不同目录
            if self.task == 'train':
                if 'sharp' in r:
                    # 检查序列名称是否在有效列表中
                    sequence_name = Path(r).parts[-2]
                    if sequence_name not in self.sequences_name:
                        continue

                    logging.info(r)
                    # 收集所有清晰图像文件
                    for file_name in f:
                        if file_name.endswith((".JPEG", ".jpg", ".png")):
                            images_info.append(Path(r, file_name))
            elif self.task == 'val':
                if 'blur_gamma' in r or 'sharp' in r:
                    # 检查序列名称是否在有效列表中
                    sequence_name = Path(r).parts[-2]
                    if sequence_name not in self.sequences_name:
                        continue

                    logging.info(r)
                    # 收集所有模糊和清晰图像文件
                    for file_name in f:
                        if file_name.endswith((".JPEG", ".jpg", ".png")):
                            images_info.append(Path(r, file_name))

        # 随机打乱图像顺序
        src_idx = np.random.permutation(len(np.asarray(images_info)))
        images_info = np.asarray(images_info)[src_idx]
        return images_info

    def init_dataset(self):
        """初始化数据集
        加载序列名称和图像路径
        Returns:
            tuple: (数据集大小, 图像路径列表)
        """
        # 加载序列名称和图像路径
        self.get_sequence_names()
        self.images_paths = self.get_image_paths()
        logging.info("Get {} images from {}".format(
            len(self.images_paths), self.dataset_cfg['images_path']))
        
        return len(self.images_paths), self.images_paths


    def __getitem__(self, index):
        patch_size = self.dataset_cfg['patch_size']
        self.counter = 0
        image_path = self.images_paths[index]

        incorrect_patch = True
        counter = -1
        while incorrect_patch:
            counter += 1
            incorrect_h = True

            while incorrect_h:
                src_BGR = dataset_utils.read_bgr_image(str(image_path))
                src_RGB = dataset_utils.bgr_to_rgb(src_BGR)
                sharp_RGB = src_RGB.copy()
                source_shape = src_RGB.shape
                h = dataset_utils.generate_homography(source_shape, self.config['homographic'])

                inv_h = np.linalg.inv(h)
                inv_h = inv_h / inv_h[2, 2]

                if self.dataset_name == 'GOPRO_ours':
                    label_path = Path(self.dataset_cfg['labels_path'], image_path.parts[-3],
                                      "{}.npz".format(image_path.stem))

                src_label = np.load(label_path, allow_pickle=True)['pts']

                dst_RGB = dataset_utils.apply_homography_to_source_image(src_RGB, inv_h)
                sharp_dst_RGB=dst_RGB
                if dst_RGB.max() > 0.0:
                    incorrect_h = False

            # 在这里添加翻转增强 =========
            if self.task == 'train':  # 只在训练时进行数据增强
                src_RGB_tensor=torch.from_numpy(src_RGB).permute(2, 0, 1).float() / 255.0
                dst_RGB_tensor = torch.from_numpy(dst_RGB).permute(2, 0, 1).float() / 255.0

                # 应用随机模糊
                blur_aug = BlurAugmentation()
                src_RGB_tensor = blur_aug.apply_blur_without_sharp(src_RGB_tensor)
                dst_RGB_tensor = blur_aug.apply_blur_without_sharp(dst_RGB_tensor)

                # 转换回numpy格式
                if src_RGB_tensor.dim() == 4:
                    src_RGB_tensor = src_RGB_tensor.squeeze(0)
                if dst_RGB_tensor.dim() == 4:
                    dst_RGB_tensor = dst_RGB_tensor.squeeze(0)
                src_RGB = src_RGB_tensor.permute(1, 2, 0).numpy() * 255.0
                dst_RGB = dst_RGB_tensor.permute(1, 2, 0).numpy() * 255.0


                dst_RGB = dataset_utils.rgb_photometric_distortion(dst_RGB)
                dst_RGB = dataset_utils.add_noise(dst_RGB, p=0.1)  # xxz_add
                dst_RGB = dataset_utils.add_shadow(dst_RGB, p=0.1)  # xxz_add

                # 随机水平翻转
                if np.random.random() > 0.33:
                    src_RGB = src_RGB[:, ::-1, :]
                    dst_RGB = dst_RGB[:, ::-1, :]
                    sharp_RGB = sharp_RGB[:, ::-1, :]
                    sharp_dst_RGB=sharp_dst_RGB[:, ::-1, :]
                    src_label[:, 0] = src_RGB.shape[1] - src_label[:, 0]

                # 随机竖直翻转
                if np.random.random() > 0.33:
                    src_RGB = src_RGB[::-1, :, :]
                    dst_RGB = dst_RGB[::-1, :, :]
                    sharp_RGB = sharp_RGB[::-1, :, :]
                    sharp_dst_RGB = sharp_dst_RGB[::-1, :, :]
                    src_label[:, 1] = src_RGB.shape[0] - src_label[:, 1]
                # 随机旋转 (-30到30度之间)
                if np.random.random() > 0.33:
                    angle = np.random.uniform(-30, 30)
                    if src_label.shape[0] == 0:  # 如果没有关键点，跳过旋转
                        continue

                    src_RGB, src_label = dataset_utils.rotate_image_and_points(src_RGB, src_label, angle)
                    if len(src_label) == 0:  # 如果所有点都被裁剪掉了，跳过这次旋转
                        continue
                    dst_RGB = dataset_utils.rotate_image_and_points(dst_RGB, None, angle)[0]
                    sharp_RGB = dataset_utils.rotate_image_and_points(sharp_RGB, None, angle)[0]
                    sharp_dst_RGB = dataset_utils.rotate_image_and_points(sharp_dst_RGB, None, angle)[0]
            else:
                dst_RGB = src_RGB

            src_label_k_best = dataset_utils.select_k_best(src_label, self.config['top_k'])
            src_heatmap = dataset_utils.labels_to_heatmap(src_label_k_best, source_shape)

            inv_h_tensor = torch.tensor(inv_h, dtype=torch.float32)
            dst_heatmap_tensor = dataset_utils.apply_homography_to_source_labels_torch(src_label_k_best, source_shape,
                                                                                       inv_h_tensor, bilinear=True)
            dst_heatmap = dst_heatmap_tensor.squeeze().numpy()

            src_RGB_norm = src_RGB / 255.0
            dst_RGB_norm = dst_RGB / 255.0
            sharp_RGB_norm= sharp_RGB  / 255.0
            sharp_dst_RGB_norm=sharp_dst_RGB / 255.0
            point_src = dataset_utils.get_window_point(source_shape, patch_size)

            im_src_patch = src_RGB_norm[int(point_src[0] - patch_size / 2): int(point_src[0] + patch_size / 2),
                           int(point_src[1] - patch_size / 2): int(point_src[1] + patch_size / 2),
                           :]
            sharp_patch = sharp_RGB_norm[int(point_src[0] - patch_size / 2): int(point_src[0] + patch_size / 2),
                           int(point_src[1] - patch_size / 2): int(point_src[1] + patch_size / 2),
                           :]

            point_dst = inv_h.dot([point_src[1], point_src[0], 1.0])
            point_dst = [point_dst[1] / point_dst[2], point_dst[0] / point_dst[2]]

            if (point_dst[0] - patch_size / 2) < 0 or (point_dst[1] - patch_size / 2) < 0:
                continue
            if (point_dst[0] + patch_size / 2) > source_shape[0] or (point_dst[1] + patch_size / 2) > source_shape[1]:
                continue

            h_src_translation = np.asanyarray([[1., 0., -(int(point_src[1]) - patch_size / 2)],
                                               [0., 1., -(int(point_src[0]) - patch_size / 2)],
                                               [0., 0., 1.]])
            h_dst_translation = np.asanyarray([[1., 0., int(point_dst[1] - patch_size / 2)],
                                               [0., 1., int(point_dst[0] - patch_size / 2)],
                                               [0., 0., 1.]])

            im_dst_patch = dst_RGB_norm[int(point_dst[0] - patch_size / 2): int(point_dst[0] + patch_size / 2),
                           int(point_dst[1] - patch_size / 2): int(point_dst[1] + patch_size / 2),
                           :]
            sharp_dst_patch=sharp_dst_RGB_norm[int(point_dst[0] - patch_size / 2): int(point_dst[0] + patch_size / 2),
                           int(point_dst[1] - patch_size / 2): int(point_dst[1] + patch_size / 2),
                           :]

            heatmap_src_patch = src_heatmap[int(point_src[0] - patch_size / 2): int(point_src[0] + patch_size / 2),
                                int(point_src[1] - patch_size / 2): int(point_src[1] + patch_size / 2)]
            heatmap_dst_patch = dst_heatmap[int(point_dst[0] - patch_size / 2): int(point_dst[0] + patch_size / 2),
                                int(point_dst[1] - patch_size / 2): int(point_dst[1] + patch_size / 2)]

            if im_src_patch.shape[0] == patch_size and im_src_patch.shape[1] == patch_size and \
                    im_dst_patch.shape[0] == patch_size and im_dst_patch.shape[1] == patch_size and \
                    heatmap_src_patch.shape[0] == patch_size and heatmap_src_patch.shape[1] == patch_size and \
                    heatmap_dst_patch.shape[0] == patch_size and heatmap_dst_patch.shape[1] == patch_size:
                incorrect_patch = False

        if incorrect_patch == False:
            homography = np.dot(h_src_translation, np.dot(h, h_dst_translation))

            homography_dst_2_src = homography.astype('float32')
            homography_dst_2_src = homography_dst_2_src / homography_dst_2_src[2, 2]

            homography_src_2_dst = np.linalg.inv(homography)
            homography_src_2_dst = homography_src_2_dst.astype('float32')
            homography_src_2_dst = homography_src_2_dst / homography_src_2_dst[2, 2]

            im_src_patch = torch.tensor(im_src_patch, dtype=torch.float32)
            im_dst_patch = torch.tensor(im_dst_patch, dtype=torch.float32)
            sharp_patch=torch.tensor(sharp_patch, dtype=torch.float32)
            sharp_dst_patch=torch.tensor(sharp_dst_patch, dtype=torch.float32)
            heatmap_src_patch = torch.tensor(heatmap_src_patch, dtype=torch.float32)
            heatmap_dst_patch = torch.tensor(heatmap_dst_patch, dtype=torch.float32)
            homography_src_2_dst = torch.tensor(homography_src_2_dst, dtype=torch.float32)
            homography_dst_2_src = torch.tensor(homography_dst_2_src, dtype=torch.float32)

            return im_src_patch.permute(2, 0, 1), im_dst_patch.permute(2, 0, 1), heatmap_src_patch.unsqueeze(
                0), heatmap_dst_patch.unsqueeze(0), homography_src_2_dst, homography_dst_2_src, sharp_patch.permute(2, 0, 1),sharp_dst_patch.permute(2, 0, 1)
