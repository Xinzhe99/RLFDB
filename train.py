import os
import datetime
import time
import tqdm
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from configs import config
from utils import common_utils, train_utils, test_utils
from utils.logger import logger
from datasets import create_dataloader
import torch.nn as nn
import numpy as np
from hsequeces_bench_kpts import compute_hsequences_metrics
from model.network import RLFDB
torch.autograd.set_detect_anomaly(True)

# Basic setting
args, cfg = config.parse_config()
start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Check directories
output_dir = Path(args.log_dir, args.exper_name, start_time)
benchmark_test_save_dir = os.path.join(output_dir, 'benchmark_test')
common_utils.check_directory(output_dir)
common_utils.check_directory('data')
ckpt_dir = output_dir / 'ckpt'
common_utils.check_directory(ckpt_dir)

# Create logger
logger.initialize(args, output_dir)

# Set random seeds
if args.fix_random_seed:
    logger.info(('Fix random seed as {}. '.format(args.random_seed)))
    common_utils.set_random_seed(args.random_seed)

tensorboard_log = SummaryWriter(common_utils.get_writer_path(args.exper_name, start_time))

# Create dataset
train_dataloaders = create_dataloader.build_dataloaders(cfg['data'], task='train', is_debugging=args.is_debugging)
val_dataloaders = create_dataloader.build_dataloaders(cfg['data'], task='val', is_debugging=args.is_debugging)

# 设置设备
device, is_parallel = train_utils.setup_device(args.gpu_ids)
print('Training device used:', device)
print('Using parallel:', is_parallel)

# 加载模型并移动到指定设备
try:
    model=RLFDB(dims=[32, 64, 128, 256], layers=[2, 2, 6, 2], expand_ratio=3.0, mlp_ratio=3.0, use_dw=True,
          drop_path_rate=0.05).to(device)
    model.to(device)

    # 如果使用并行，需要封装模型
    if is_parallel:
        gpu_list = [int(i) for i in args.gpu_ids.split(',')]
        print(f"Using DataParallel for multi-GPU training, GPU list: {gpu_list}")
        model = nn.DataParallel(model, device_ids=gpu_list)
except Exception as e:
    print(f"Error occurred during model loading or device transfer: {str(e)}")
    raise

optimizer = train_utils.build_optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                        cfg['model']['optimizer'])

total_epochs = cfg['model']['optimizer']['total_epochs']
start_epoch = 0
last_epoch = -1
best_eval_loss_epoch = 0
best_eval_loss = float('inf')
best_repeatability_epoch = 0
best_repeatability = -float('inf')

#benchmark测评用
best_benchmark_repeatability_epoch = 0
best_benchmark_repeatability = -float('inf')

## Count the number of learnable parameters.
logger.info("================ List of Learnable model parameters ================ ")
for n, p in model.named_parameters():
    if p.requires_grad:
        logger.info("{} {}".format(n, p.data.shape))
    else:
        logger.info("\n\n\n None learnable params {} {}".format(n, p.data.shape))
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
logger.info("The number of learnable parameters : {} ".format(params.data))
logger.info("==================================================================== ")

scheduler = train_utils.build_scheduler(
    optimizer,
    total_epochs=total_epochs,
    last_epoch=last_epoch,
    scheduler_cfg=cfg['model']['scheduler'])

with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True) as tbar:
    for cur_epoch in tbar:
        # 训练部分保持不变
        for train_loader in train_dataloaders:
            loss = train_utils.train_model(
                cur_epoch=cur_epoch, dataloader=train_loader['dataloader'], model=model, optimizer=optimizer,
                device=device, tb_log=tensorboard_log, tbar=tbar, output_dir=output_dir,
                cell_size=cfg['model']['cell_size'],
                anchor_loss=cfg['model']['anchor_loss'], usp_loss=None, repeatability_loss=None
            )

        if cur_epoch > 0 and (cur_epoch + 1) % cfg['ckpt_save_interval'] == 0:
            # 保存当前epoch的模型
            ckpt_name = ckpt_dir / ('checkpoint_epoch_%d' % cur_epoch)
            logger.save_model(train_utils.ckpt_state(model, optimizer, cur_epoch, 0), ckpt_name)# Save model

            with torch.no_grad():# Also perform evaluation during training, step1: inference to get npy
                model.eval()
                logger.info("Start evaluation on Hsequence dataset:")
                t_start_bench=time.time()
                # Ensure directory exists
                os.makedirs(benchmark_test_save_dir, exist_ok=True)
                test_predict_result_path = os.path.join(benchmark_test_save_dir, f'epoch_{cur_epoch}')
                # Ensure directory exists
                os.makedirs(os.path.dirname(test_predict_result_path), exist_ok=True)
                # Traverse all subfolders
                for subdir in tqdm.tqdm(os.listdir(args.benchmark_input_root)):
                    input_subdir = os.path.join(args.benchmark_input_root, subdir)
                    output_subdir = os.path.join(test_predict_result_path, subdir)
                    if not os.path.isdir(input_subdir):
                        continue
                    # Create corresponding output subdirectory
                    os.makedirs(output_subdir, exist_ok=True)
                    # Process each .ppm file
                    for filename in os.listdir(input_subdir):
                        if filename.endswith('.ppm'):
                            input_path = os.path.join(input_subdir, filename)
                            output_path = os.path.join(output_subdir, f"{filename}.kpt.npy")
                            # Load image and detect keypoints
                            im_rgb = train_utils.load_im(input_path)
                            keypoints = test_utils.detect(args, im_rgb, model, device)
                            # Save keypoints
                            np.save(output_path, keypoints)

                #step2：对得到的npy文件进行重复率计算
                args_hsequences_metrics=config.parse_hsequences_metrics()
                args_hsequences_metrics.results_dir=test_predict_result_path
                args_hsequences_metrics.results_bench_dir=benchmark_test_save_dir
                args_hsequences_metrics.detector_name=str(cur_epoch)
                test_results = compute_hsequences_metrics(args_hsequences_metrics)
                if test_results:
                    logger.info(f"\nEpoch {cur_epoch} - Hsequences_metrics results:")
                    for key, value in test_results.items():
                        if key != 'raw_results':  # Optionally skip printing raw results
                            logger.info(f"  {key}: {value}")
                rep_single = test_results.get('repeatability_single_scale')
                if rep_single > best_benchmark_repeatability:
                    best_benchmark_repeatability = rep_single
                    best_benchmark_repeatability_epoch = cur_epoch
                    logger.info(
                        f"Best single-scale repeatability improved to {best_benchmark_repeatability:.4f} at epoch {best_benchmark_repeatability_epoch}")
                else:
                    logger.info(
                        f"Current single-scale repeatability ({rep_single:.4f}) did not exceed the best value ({best_benchmark_repeatability:.4f}, achieved at epoch {best_benchmark_repeatability_epoch})")
                t_end_bench = time.time()
                print('Evaluation on benchmark dataset took:', (t_end_bench - t_start_bench) / 60, 'minutes')
        scheduler.step()

logger.info(f"Best rep_s_nms on Hsequence Benchmark: {best_benchmark_repeatability:.4f} (Epoch {best_benchmark_repeatability_epoch})")
logger.info("================ Training finished ================ \n\n")