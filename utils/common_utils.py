import random
import yaml
import numpy as np
from pathlib import Path
import importlib
import torch


def get_cfg_from_yaml_file(cfg_file):
    """从YAML文件中加载配置
    Args:
        cfg_file: YAML配置文件路径
    Returns:
        config: 配置字典
    """
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            # 如果FullLoader不可用，回退到基础loader
            config = yaml.load(f)

    return config

def set_random_seed(seed):
    """设置随机种子以确保实验可重复性
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 禁用CUDNN的随机性，确保结果可复现
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_directory(file_path):
    """检查并创建目录
    Args:
        file_path: 目录路径
    """
    file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)

def get_writer_path(exper_name, start_time):
    """获取TensorBoard日志写入路径
    Args:
        exper_name: 实验名称
        start_time: 开始时间
    Returns:
        str: 完整的日志路径
    """
    output_dir = Path('runs')
    return str(output_dir / exper_name / start_time)


def get_module(path, name):
    """动态导入模块
    Args:
        path: 模块路径
        name: 模块名称
    Returns:
        module: 导入的模块对象
    """
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)


def remove_borders(images, borders=3):
    """移除图像边界
    Args:
        images: 输入图像张量，支持4D [B,C,H,W]、3D [C,H,W] 或 2D [H,W] 格式
        borders: 要移除的边界宽度，默认为3
    Returns:
        images: 处理后的图像张量
    """
    shape = images.shape

    if len(shape) == 4:
        # 处理4D张量 [B,C,H,W]
        for batch_id in range(shape[0]):
            images[batch_id, :, 0:borders, :] = 0
            images[batch_id, :, :, 0:borders] = 0
            images[batch_id, :, shape[2] - borders:shape[2], :] = 0
            images[batch_id, :, :, shape[3] - borders:shape[3]] = 0
    elif len(shape) == 3:
        # 处理3D张量 [C,H,W]
        images[:, 0:borders, :] = 0
        images[:, :, 0:borders] = 0
        images[:, shape[1] - borders:shape[1], :] = 0
        images[:, :, shape[2] - borders:shape[2]] = 0
    elif len(shape) == 2:
        # 处理2D张量 [H,W]
        images[0:borders, :] = 0
        images[:, 0:borders] = 0
        images[shape[0] - borders:shape[0], :] = 0
        images[:, shape[1] - borders:shape[1]] = 0
    else:
        raise ValueError(f"Unsupported tensor shape: {shape}. Expected 2D, 3D or 4D tensor.")

    return images


def create_result_dir(path):
    """创建结果目录结构
    Args:
        path: 目录路径字符串
    """
    # 将路径字符串转换为Path对象进行处理
    path_obj = Path(path)
    # 逐级创建父目录
    current_path = Path()
    for part in path_obj.parts[:-1]:
        current_path = current_path / part
        check_directory(current_path)