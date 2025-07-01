import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
import random


class DiskBlur(nn.Module):
    """散焦模糊模块，使用圆盘卷积核实现"""

    def __init__(self, radius: int, p: float = 1.0):
        super().__init__()
        if not isinstance(radius, int) or radius <= 0:
            raise ValueError("Radius must be a positive integer.")
        if not isinstance(p, float) or not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be a float between 0.0 and 1.0.")
        self.radius = radius
        self.p = p
        self.kernel = self._create_disk_kernel(radius)
        self.pad = nn.ReflectionPad2d(radius)

    def _create_disk_kernel(self, radius: int) -> torch.Tensor:
        kernel_size = 2 * radius + 1
        center = radius
        kernel = torch.zeros(kernel_size, kernel_size)
        y, x = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing='ij')
        distance = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        mask = distance <= radius
        kernel[mask] = 1.0
        kernel = kernel / torch.sum(kernel)
        return kernel.float()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        if img.ndim < 3:
            raise ValueError(f"Expected input tensor to have at least 3 dimensions (C, H, W), but got {img.ndim}")

        is_batch = img.ndim == 4
        if not is_batch:
            img = img.unsqueeze(0)

        batch_size, num_channels, height, width = img.shape
        device = img.device

        disk_kernel = self.kernel.to(device)
        conv_kernel = disk_kernel.repeat(num_channels, 1, 1, 1)
        img_padded = self.pad(img)
        blurred_img = F.conv2d(img_padded, conv_kernel, padding=0, groups=num_channels)

        if not is_batch:
            blurred_img = blurred_img.squeeze(0)

        return blurred_img


class BlurAugmentation:
    """图像模糊增强类，包含多种模糊效果"""

    def __init__(self):
        # 运动模糊配置 - 进一步增强参数
        self.motion_blur_weak = K.RandomMotionBlur(
            kernel_size=(15, 15),  # 更大的kernel_size
            angle=(0, 180),
            direction=(-1.0, 1.0),  # 最大方向范围
            p=1.0
        )
        self.motion_blur_medium = K.RandomMotionBlur(
            kernel_size=(21, 21),  # 显著增加kernel_size
            angle=(0, 180),
            direction=(-1.0, 1.0),
            p=1.0
        )
        self.motion_blur_strong = K.RandomMotionBlur(
            kernel_size=(31, 31),  # 极大的kernel_size
            angle=(0, 180),
            direction=(-1.0, 1.0),
            p=1.0
        )

        # 高斯模糊配置 - 进一步增强参数
        self.gaussian_blur_weak = K.RandomGaussianBlur(
            kernel_size=(11, 11),  # 增加kernel_size
            sigma=(3.0, 4.5),  # 增加sigma范围
            p=1.0
        )
        self.gaussian_blur_medium = K.RandomGaussianBlur(
            kernel_size=(15, 15),
            sigma=(4.5, 6.5),
            p=1.0
        )
        self.gaussian_blur_strong = K.RandomGaussianBlur(
            kernel_size=(21, 21),  # 显著增加kernel_size
            sigma=(6.5, 9.0),  # 显著增加sigma范围
            p=1.0
        )

        # 保持散焦模糊配置不变
        self.disk_blur_weak = DiskBlur(radius=5, p=1.0)
        self.disk_blur_medium = DiskBlur(radius=9, p=1.0)
        self.disk_blur_strong = DiskBlur(radius=13, p=1.0)

    def apply_blur_with_sharp(self, image):
        """随机应用一种模糊效果"""
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a torch.Tensor")

        if image.ndim == 3:
            image = image.unsqueeze(0)

        choice = random.random()
        strength = random.random()

        if choice < 0.25:  # 保持不变
            return image
        elif choice < 0.5:  # 高斯模糊
            if strength < 0.33:
                return self.gaussian_blur_weak(image)
            elif strength < 0.67:
                return self.gaussian_blur_medium(image)
            else:
                return self.gaussian_blur_strong(image)
        elif choice < 0.75:  # 散焦模糊
            if strength < 0.33:
                return self.disk_blur_weak(image)
            elif strength < 0.67:
                return self.disk_blur_medium(image)
            else:
                return self.disk_blur_strong(image)
        else:  # 运动模糊
            if strength < 0.33:
                return self.motion_blur_weak(image)
            elif strength < 0.67:
                return self.motion_blur_medium(image)
            else:
                return self.motion_blur_strong(image)

    def apply_blur_without_sharp(self, image):
        """随机应用模糊效果，去掉不模糊的选项"""
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a torch.Tensor")

        if image.ndim == 3:
            image = image.unsqueeze(0)

        choice = random.random()
        strength = random.random()

        if choice < 0.33:  # 高斯模糊
            if strength < 0.33:
                return self.gaussian_blur_weak(image)
            elif strength < 0.67:
                return self.gaussian_blur_medium(image)
            else:
                return self.gaussian_blur_strong(image)
        elif choice < 0.67:  # 散焦模糊
            if strength < 0.33:
                return self.disk_blur_weak(image)
            elif strength < 0.67:
                return self.disk_blur_medium(image)
            else:
                return self.disk_blur_strong(image)
        else:  # 运动模糊
            if strength < 0.33:
                return self.motion_blur_weak(image)
            elif strength < 0.67:
                return self.motion_blur_medium(image)
            else:
                return self.motion_blur_strong(image)
