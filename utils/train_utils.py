import os
import logging
import time
import numpy as np
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision
from benchmark_test import repeatability_tools, geometry_tools
from loss import loss_function
import torch.nn as nn
from datasets import dataset_utils



def build_optimizer(params, optim_cfg):
    logging.info("=> Setting %s solver", optim_cfg['name'])

    if optim_cfg['name'] == 'adam':
        optimizer = optim.Adam(params, lr=optim_cfg['lr'], weight_decay=optim_cfg['weight_decay'])
    elif optim_cfg['name'] == 'sgd':
        optimizer = optim.SGD(params, lr=optim_cfg['lr'])
    elif optim_cfg['name'] == 'adamw':
        optimizer = optim.AdamW(params, lr=optim_cfg['lr'], weight_decay=optim_cfg['weight_decay'])
    else:
        raise ValueError("Optimizer [%s] not recognized." % optim_cfg['name'])
    return optimizer


def build_scheduler(optimizer, total_epochs, last_epoch, scheduler_cfg):
    logging.info("=> Setting %s scheduler", scheduler_cfg['name'])
    if scheduler_cfg['name'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         patience=scheduler_cfg['patience'],
                                                         factor=scheduler_cfg['factor'],
                                                         min_lr=scheduler_cfg['min_lr'])
    elif scheduler_cfg['name'] == 'sgdr':
        # Assuming WarmRestart is a custom implementation, you'd keep it as is
        scheduler = WarmRestart(optimizer)
    elif scheduler_cfg['name'] == 'linear':
        # Assuming LinearDecay is a custom implementation, you'd keep it as is
        scheduler = LinearDecay(optimizer,
                                min_lr=scheduler_cfg['min_lr'],
                                max_epoch=total_epochs,
                                start_epoch=scheduler_cfg['start_epoch'],
                                last_epoch=last_epoch)
    elif scheduler_cfg['name'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=total_epochs,
                                      eta_min=0)
    else:
        raise ValueError("Scheduler [%s] not recognized." % scheduler_cfg['name'])
    return scheduler

class WarmRestart(optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max=30, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for
                base_lr in self.base_lrs]


class LinearDecay(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epoch, start_epoch=0, min_lr=0, last_epoch=-1):
        self.max_epoch = max_epoch
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.start_epoch:
            return self.base_lrs
        return [base_lr - ((base_lr - self.min_lr) / self.max_epoch) * (self.last_epoch - self.start_epoch) for
                base_lr in self.base_lrs]


def train_model(
        cur_epoch, dataloader, model, optimizer, device, tb_log,
        tbar, output_dir=None, cell_size=8, add_dustbin=True, # add_dustbin 未使用，可移除
        anchor_loss='softmax', usp_loss=None, repeatability_loss=None): # anchor_loss, usp_loss, repeatability_loss 未使用，可移除

    total_loss_avg = []
    disp_dict = {}

    tic = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for idx, batch in pbar:
        global_step = cur_epoch * len(dataloader) + idx

        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch, sharp_patch,sharp_dst_patch= batch
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch, sharp_patch,sharp_dst_patch = \
            images_src_batch.to(device), images_dst_batch.to(device), heatmap_src_patch.to(
                device), heatmap_dst_patch.to(device), h_src_2_dst_batch.to(device), h_dst_2_src_batch.to(device), sharp_patch.to(device),sharp_dst_patch.to(device)

        try:
            cur_lr = float(optimizer.lr) # optimizer 没有 .lr 属性
        except AttributeError:
            cur_lr = optimizer.param_groups[0]['lr']

        tb_log.add_scalar('learning_rate', cur_lr, global_step)

        model.train()

        src_outputs = model(images_src_batch)
        dst_outputs = model(images_dst_batch)
        sharp_src_outputs = model(sharp_patch)
        sharp_dst_outputs=model(sharp_dst_patch)

        #该反映模糊图片的探测准确度
        src_anchor_loss = loss_function.detector_loss(
            keypoint_map=heatmap_src_patch,
            logits=src_outputs['logits'],
            device=device,
            grid_size=cell_size
        )
        # 该反映清晰图片的探测准确度
        sharp_anchor_loss = loss_function.detector_loss(
            keypoint_map=heatmap_src_patch,
            logits=sharp_src_outputs['logits'],
            device=device,
            grid_size=cell_size
        )
        dst_anchor_loss = loss_function.detector_loss(
            keypoint_map=heatmap_dst_patch,
            logits=dst_outputs['logits'],
            device=device,
            grid_size=cell_size
        )
        sharp_dst_anchor_loss = loss_function.detector_loss(
            keypoint_map=heatmap_dst_patch,
            logits=sharp_dst_outputs['logits'],
            device=device,
            grid_size=cell_size
        )

        load_balancing_loss=src_outputs['load_balancing_loss'].mean()+dst_outputs['load_balancing_loss'].mean()

        # loss = src_anchor_loss + dst_anchor_loss + sharp_anchor_loss + sharp_dst_anchor_loss+ align_loss + load_balancing_loss
        loss = src_anchor_loss + dst_anchor_loss + sharp_anchor_loss + sharp_dst_anchor_loss + load_balancing_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_avg.append(loss.item())

        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        pbar.update()
        avg_loss_val = np.mean(total_loss_avg) if total_loss_avg else 0 # 计算平均损失
        # pbar.set_description(
        #     'Training current lr: {:0.5f}, Total loss: {:0.4f}, src_anchor_loss: {:0.4f}, dst_anchor_loss: {:0.4f}, sharp_anchor_loss: {:0.4f},sharp_dst_anchor_loss: {:0.4f}, load_balancing_loss : {:0.4f}, align_loss : {:0.4f}, avg loss: {:0.4f}'
        #     .format(cur_lr, loss.item(), src_anchor_loss.item(), dst_anchor_loss.item(), sharp_anchor_loss.item(),sharp_dst_anchor_loss.item(),
        #             load_balancing_loss.item(), align_loss.item(), avg_loss_val))
        pbar.set_description(
            'Training current lr: {:0.5f}, Total loss: {:0.4f}, src_anchor_loss: {:0.4f}, dst_anchor_loss: {:0.4f}, sharp_anchor_loss: {:0.4f},sharp_dst_anchor_loss: {:0.4f}, load_balancing_loss : {:0.4f},avg loss: {:0.4f}'
            .format(cur_lr, loss.item(), src_anchor_loss.item(), dst_anchor_loss.item(), sharp_anchor_loss.item(),
                    sharp_dst_anchor_loss.item(), load_balancing_loss.item(), avg_loss_val))
        tbar.set_postfix(disp_dict)
        tbar.refresh()

        if idx % 50 == 0:
            tb_log.add_scalar('train/total_loss', loss.item(), global_step)
            tb_log.add_scalar('train/src_anchor_loss', src_anchor_loss.item(), global_step)
            tb_log.add_scalar('train/dst_anchor_loss', dst_anchor_loss.item(), global_step)
            tb_log.add_scalar('train/sharp_anchor_loss', sharp_anchor_loss.item(), global_step)
            tb_log.add_scalar('train/sharp_dst_anchor_loss', sharp_dst_anchor_loss.item(), global_step)
            tb_log.add_scalar('train/load_balancing_loss', load_balancing_loss.item(), global_step)
            # tb_log.add_scalar('train/align_loss', align_loss.item(), global_step)

            if idx == 0:
                src_image_grid = torchvision.utils.make_grid(images_src_batch)
                dst_image_grid = torchvision.utils.make_grid(images_dst_batch)
                tb_log.add_image('train/src_image', src_image_grid, global_step)
                tb_log.add_image('train/dst_image', dst_image_grid, global_step)

                src_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_src_patch)
                dst_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_dst_patch)
                tb_log.add_image('train/src_input_heatmap', src_input_heatmaps_grid, global_step)
                tb_log.add_image('train/dst_input_heatmap', dst_input_heatmaps_grid, global_step)

                src_outputs_score_maps = prob_to_score_maps_tensor_batch(src_outputs['prob'], cell_size)
                dst_outputs_score_maps = prob_to_score_maps_tensor_batch(dst_outputs['prob'], cell_size)
                src_outputs_score_maps_grid = torchvision.utils.make_grid(src_outputs_score_maps)
                dst_outputs_score_maps_grid = torchvision.utils.make_grid(dst_outputs_score_maps)
                tb_log.add_image('train/src_output_heatmap', src_outputs_score_maps_grid, global_step)
                tb_log.add_image('train/dst_output_heatmap', dst_outputs_score_maps_grid, global_step)

    toc = time.time()
    # total_loss_avg = torch.stack(total_loss_avg) # 现在是 list of floats
    final_avg_loss = np.mean(total_loss_avg) if total_loss_avg else 0
    logging.info(
        'Epoch {} (Training). Loss: {:0.4f}. Time per epoch: {}. \n'.format(cur_epoch, final_avg_loss,
                                                                            round(toc - tic, 4)))

    return final_avg_loss # 返回平均损失值

def prob_to_score_maps_tensor_batch(prob_outputs, nms_size):
    score_maps_np = prob_outputs.detach().cpu().numpy()

    score_maps_nms_batch = [repeatability_tools.apply_nms_fast(h, nms_size) for h in score_maps_np]
    score_maps_nms_batch = np.stack(score_maps_nms_batch, axis=0)
    score_maps_nms_batch_tensor = torch.tensor(score_maps_nms_batch, dtype=torch.float32)
    return score_maps_nms_batch_tensor.unsqueeze(1)

def compute_repeatability_with_maximum_filter(src_scores_np, dst_scores_np, homography, mask_src, mask_dst, nms_size,
                                              num_points):
    """
    计算两幅图像间特征点检测的重复性指标，使用最大值滤波进行非极大值抑制
    
    Args:
        src_scores_np: 源图像的特征响应图
        dst_scores_np: 目标图像的特征响应图
        homography: 从目标图像到源图像的单应性变换矩阵
        mask_src: 源图像的有效区域掩码
        mask_dst: 目标图像的有效区域掩码
        nms_size: 非极大值抑制的窗口大小
        num_points: 每张图像要提取的特征点数量
    
    Returns:
        rep_s_nms: 单尺度重复性得分
        rep_m_nms: 多尺度重复性得分
        error_overlap_s_nms: 单尺度重叠误差
        error_overlap_m_nms: 多尺度重叠误差
        possible_matches_nms: 可能的匹配点对数量
    """
    rep_s_nms = []
    rep_m_nms = []
    error_overlap_s_nms = []
    error_overlap_m_nms = []
    possible_matches_nms = []

    src_scores_nms = repeatability_tools.apply_nms(src_scores_np, nms_size)
    dst_scores_nms = repeatability_tools.apply_nms(dst_scores_np, nms_size)

    src_scores_common_nms = np.multiply(src_scores_nms, mask_src)
    dst_scores_common_nms = np.multiply(dst_scores_nms, mask_dst)

    src_pts_nms = geometry_tools.get_point_coordinates(src_scores_common_nms, num_points=num_points, order_coord='xysr')
    dst_pts_nms = geometry_tools.get_point_coordinates(dst_scores_common_nms, num_points=num_points, order_coord='xysr')

    dst_to_src_pts_nms = geometry_tools.apply_homography_to_points(dst_pts_nms, homography)

    repeatability_results_nms = repeatability_tools.compute_repeatability(src_pts_nms, dst_to_src_pts_nms)

    rep_s_nms.append(repeatability_results_nms['rep_single_scale'])
    rep_m_nms.append(repeatability_results_nms['rep_multi_scale'])
    error_overlap_s_nms.append(repeatability_results_nms['error_overlap_single_scale'])
    error_overlap_m_nms.append(repeatability_results_nms['error_overlap_multi_scale'])
    possible_matches_nms.append(repeatability_results_nms['possible_matches'])

    return rep_s_nms, rep_m_nms, error_overlap_s_nms, error_overlap_m_nms, possible_matches_nms


# def ckpt_state(model=None, optimizer=None, epoch=None, rep_s=0.):
#     optim_state = optimizer.state_dict() if optimizer is not None else optimizer
#     model_state = model.state_dict() if model is not None else None
#
#     return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'repeatability': rep_s}
#todo revised 为了并行训练
def ckpt_state(model=None, optimizer=None, epoch=None, rep_s=0.):
    optim_state = optimizer.state_dict() if optimizer is not None else optimizer

    if model is not None:
        # 获取原始状态字典
        model_state_dict = model.state_dict()
        # 创建一个新的状态字典，有条件地移除"module."前缀
        new_model_state_dict = {}

        # 检查是否存在以"module."开头的键
        has_module_prefix = any(k.startswith('module.') for k in model_state_dict.keys())

        # 只有当存在"module."前缀时才进行处理
        for k, v in model_state_dict.items():
            if has_module_prefix and k.startswith('module.'):
                name = k[7:]  # 移除'module.'前缀
            else:
                name = k  # 保持原样
            new_model_state_dict[name] = v
        model_state = new_model_state_dict
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'repeatability': rep_s}

@torch.no_grad()
def check_val_repeatability(dataloader, model, device, tb_log, cur_epoch, cell_size=8, nms_size=15, num_points=25):
    rep_s = []
    rep_m = []
    error_overlap_s = []
    error_overlap_m = []
    possible_matches = []
    disp_dict = {}

    #xxz debug#########
    rep_s_nms_list = []
    rep_m_nms_list = []
    error_overlap_s_nms_list = []
    error_overlap_m_nms_list = []
    possible_matches_nms_list = []
    ##########

    model.eval()

    iterate = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")

    ba = 0
    cb = 0
    dc = 0
    ed = 0
    fe = 0
    for idx, batch in iterate:
        a = time.time()
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch = batch
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch = \
            images_src_batch.to(device), images_dst_batch.to(device), heatmap_src_patch.to(
                device), heatmap_dst_patch.to(device), h_src_2_dst_batch.to(device), h_dst_2_src_batch.to(device)

        src_outputs = model(images_src_batch)
        dst_outputs = model(images_dst_batch)

        src_score_maps = src_outputs['prob']
        dst_score_maps = dst_outputs['prob']

        b = time.time()
        homography = h_dst_2_src_batch[0].cpu().numpy()
        shape_src = images_src_batch[0].permute(1, 2, 0).cpu().numpy().shape
        shape_dst = images_dst_batch[0].permute(1, 2, 0).cpu().numpy().shape
        mask_src, mask_dst = geometry_tools.create_common_region_masks(homography, shape_src, shape_dst)
        #- mask_src ：源图像中的有效区域（与目标图像重叠的部分）
        #- mask_dst ：目标图像中的有效区域（与源图像重叠的部分）
        c = time.time()

        src_scores = src_score_maps
        dst_scores = dst_score_maps
        # Apply NMS
        src_scores = repeatability_tools.get_nms_score_map_from_score_map(src_scores[0, :, :].cpu().numpy(),
                                                                          conf_thresh=0.015, nms_size=nms_size)
        dst_scores = repeatability_tools.get_nms_score_map_from_score_map(dst_scores[0, :, :].cpu().numpy(),
                                                                          conf_thresh=0.015, nms_size=nms_size)

        src_scores = np.multiply(src_scores, mask_src)
        dst_scores = np.multiply(dst_scores, mask_dst)

        d = time.time()

        src_pts = geometry_tools.get_point_coordinates(src_scores, num_points=num_points, order_coord='xysr')
        dst_pts = geometry_tools.get_point_coordinates(dst_scores, num_points=num_points, order_coord='xysr')

        dst_to_src_pts = geometry_tools.apply_homography_to_points(dst_pts, homography)

        e = time.time()

        repeatability_results = repeatability_tools.compute_repeatability(src_pts, dst_to_src_pts)

        rep_s.append(repeatability_results['rep_single_scale'])
        rep_m.append(repeatability_results['rep_multi_scale'])
        error_overlap_s.append(repeatability_results['error_overlap_single_scale'])
        error_overlap_m.append(repeatability_results['error_overlap_multi_scale'])
        possible_matches.append(repeatability_results['possible_matches'])

        f = time.time()

        ## time count
        ba += b - a
        cb += c - b
        dc += d - c
        ed += e - d
        fe += f - e

        iterate.update()
        iterate.set_description('Validation time: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}'.format(ba, cb, dc, ed, fe))
        disp_dict.update({'rep_s': '{:0.2f}'.format(repeatability_results['rep_single_scale'])})
        iterate.set_postfix(disp_dict)


        rep_s_nms, rep_m_nms, error_overlap_s_nms, error_overlap_m_nms, possible_matches_nms = \
            compute_repeatability_with_maximum_filter(
                src_score_maps[0, :, :].cpu().numpy(), dst_score_maps[0, :, :].cpu().numpy(), homography,
                mask_src, mask_dst, nms_size, num_points
            )
        #xxz debug#########
        rep_s_nms_list.append(rep_s_nms)
        rep_m_nms_list.append(rep_m_nms)
        error_overlap_s_nms_list.append(error_overlap_s_nms)
        error_overlap_m_nms_list.append(error_overlap_m_nms)
        possible_matches_nms_list.append(possible_matches_nms)
        ################################################
        if tb_log is not None and idx == 0:
            src_image_grid = torchvision.utils.make_grid(images_src_batch)
            dst_image_grid = torchvision.utils.make_grid(images_dst_batch)
            tb_log.add_image('val_src_image', src_image_grid, cur_epoch)
            tb_log.add_image('val_dst_image', dst_image_grid, cur_epoch)

            src_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_src_patch)
            dst_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_dst_patch)
            tb_log.add_image('val_src_input_heatmap', src_input_heatmaps_grid, cur_epoch)
            tb_log.add_image('val_dst_input_heatmap', dst_input_heatmaps_grid, cur_epoch)

            src_outputs_score_maps = prob_to_score_maps_tensor_batch(src_outputs['prob'], nms_size)
            dst_outputs_score_maps = prob_to_score_maps_tensor_batch(dst_outputs['prob'], nms_size)
            src_outputs_score_maps_grid = torchvision.utils.make_grid(src_outputs_score_maps)
            dst_outputs_score_maps_grid = torchvision.utils.make_grid(dst_outputs_score_maps)
            tb_log.add_image('val_src_output_heatmap', src_outputs_score_maps_grid, cur_epoch)
            tb_log.add_image('val_dst_output_heatmap', dst_outputs_score_maps_grid, cur_epoch)

    return np.asarray(rep_s).mean(), np.asarray(rep_m).mean(), np.asarray(error_overlap_s).mean(), \
           np.asarray(error_overlap_m).mean(), np.asarray(possible_matches).mean(), \
           np.asarray(rep_s_nms_list).mean(), np.asarray(rep_m_nms_list).mean(), np.asarray(error_overlap_s_nms_list).mean(), \
           np.asarray(error_overlap_m_nms_list).mean(), np.asarray(possible_matches_nms_list).mean()


@torch.no_grad()
def check_val_hsequences_repeatability(dataloader, model, device, tb_log, cur_epoch, cell_size=8, nms_size=15,
                                       num_points=25, border_size=15):
    rep_s = []
    rep_m = []
    error_overlap_s = []
    error_overlap_m = []
    possible_matches = []
    disp_dict = {}

    counter_sequences = 0
    iterate = tqdm(range(len(dataloader.sequences)), total=len(dataloader.sequences), desc="HSequences Eval")

    ba = 0
    cb = 0
    dc = 0
    ed = 0
    fe = 0
    for sequence_index in iterate:
        sequence_data = dataloader.get_sequence_data(sequence_index)

        counter_sequences += 1

        sequence_name = sequence_data['sequence_name']
        im_src_RGB_norm = sequence_data['im_src_RGB_norm']
        images_dst_RGB_norm = sequence_data['images_dst_RGB_norm']
        h_dst_2_src = sequence_data['h_dst_2_src']

        for im_dst_index in range(len(images_dst_RGB_norm)):
            a = time.time()
            pts_src, src_outputs_score_maps = extract_detections(
                im_src_RGB_norm, model, device,
                cell_size=cell_size, nms_size=nms_size, num_points=num_points, border_size=border_size
            )
            pts_dst, dst_outputs_score_maps = extract_detections(
                images_dst_RGB_norm[im_dst_index], model, device,
                cell_size=cell_size, nms_size=nms_size, num_points=num_points, border_size=border_size
            )

            b = time.time()

            mask_src, mask_dst = geometry_tools.create_common_region_masks(
                h_dst_2_src[im_dst_index], im_src_RGB_norm.shape, images_dst_RGB_norm[im_dst_index].shape
            )

            c = time.time()

            pts_src = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_src)))
            pts_dst = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_dst)))

            idx_src = repeatability_tools.check_common_points(pts_src, mask_src)
            if idx_src.size == 0:
                continue
            pts_src = pts_src[idx_src]

            idx_dst = repeatability_tools.check_common_points(pts_dst, mask_dst)
            if idx_dst.size == 0:
                continue
            pts_dst = pts_dst[idx_dst]

            d = time.time()

            pts_src = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_src)))
            pts_dst = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_dst)))

            pts_dst_to_src = geometry_tools.apply_homography_to_points(
                pts_dst, h_dst_2_src[im_dst_index])

            e = time.time()

            repeatability_results = repeatability_tools.compute_repeatability(pts_src, pts_dst_to_src)

            rep_s.append(repeatability_results['rep_single_scale'])
            rep_m.append(repeatability_results['rep_multi_scale'])
            error_overlap_s.append(repeatability_results['error_overlap_single_scale'])
            error_overlap_m.append(repeatability_results['error_overlap_multi_scale'])
            possible_matches.append(repeatability_results['possible_matches'])

            f = time.time()

            ## time count
            ba += b - a
            cb += c - b
            dc += d - c
            ed += e - d
            fe += f - e

            iterate.set_description("{}  {} / {} - {} Validation time: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}"
            .format(
                sequence_name, counter_sequences, len(dataloader.sequences), im_dst_index,
                ba, cb, dc, ed, fe
            ))
            disp_dict.update({'rep_s': '{:0.2f}'.format(repeatability_results['rep_single_scale'])})
            iterate.set_postfix(disp_dict)

            if tb_log is not None and sequence_index == 50:
                im_src_RGB_norm_tensor = torch.tensor(im_src_RGB_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(
                    0)
                im_dst_RGB_norm_tensor = torch.tensor(images_dst_RGB_norm[im_dst_index], dtype=torch.float32).permute(2,
                                                                                                                      0,
                                                                                                                      1).unsqueeze(
                    0)
                src_image_grid = torchvision.utils.make_grid(im_src_RGB_norm_tensor)
                dst_image_grid = torchvision.utils.make_grid(im_dst_RGB_norm_tensor)
                tb_log.add_image('val_src_image', src_image_grid, cur_epoch)
                tb_log.add_image('val_dst_image', dst_image_grid, cur_epoch)

                src_outputs_score_maps_grid = torchvision.utils.make_grid(src_outputs_score_maps)
                dst_outputs_score_maps_grid = torchvision.utils.make_grid(dst_outputs_score_maps)
                tb_log.add_image('val_src_output_heatmap', src_outputs_score_maps_grid, cur_epoch)
                tb_log.add_image('val_dst_output_heatmap', dst_outputs_score_maps_grid, cur_epoch)

    return np.asarray(rep_s).mean(), np.asarray(rep_m).mean(), np.asarray(error_overlap_s).mean(), \
        np.asarray(error_overlap_m).mean(), np.asarray(possible_matches).mean()


@torch.no_grad()
def extract_detections(image_RGB_norm, model, device, cell_size=8, nms_size=15, num_points=25, border_size=15):
    height_RGB_norm, width_RGB_norm = image_RGB_norm.shape[0], image_RGB_norm.shape[1]

    image_even = dataset_utils.make_shape_even(image_RGB_norm)
    height_even, width_even = image_even.shape[0], image_even.shape[1]

    image_pad = dataset_utils.mod_padding_symmetric(image_even, factor=64)

    image_pad_tensor = torch.tensor(image_pad, dtype=torch.float32)
    image_pad_tensor = image_pad_tensor.permute(2, 0, 1)
    image_pad_batch = image_pad_tensor.unsqueeze(0)

    with torch.no_grad():
        output_pad_batch = model(image_pad_batch.to(device))

    score_map_pad_batch = output_pad_batch['prob']
    score_map_pad_np = score_map_pad_batch[0, :, :].cpu().detach().numpy()

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]

    score_map_batch = score_map_pad_batch[:, :, h_start:h_end, w_start:w_end]

    score_map_remove_border = geometry_tools.remove_borders(score_map, borders=border_size)
    score_map_nms = repeatability_tools.apply_nms(score_map_remove_border, nms_size)

    pts = geometry_tools.get_point_coordinates(score_map_nms, num_points=num_points, order_coord='xysr')

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:num_points]

    return pts_output, score_map_batch
#todo my add
def setup_device(gpu_ids=None):
    """
    设置GPU设备

    参数:
        gpu_ids: None表示使用CPU，"0"表示使用cuda:0，"1"表示使用cuda:1，
                "0,1"表示同时使用cuda:0和cuda:1

    返回:
        device: 设备对象
        is_parallel: 是否使用并行
    """
    is_parallel = False

    if gpu_ids is None:
        # 使用CPU
        device = torch.device('cpu')
    else:
        # 设置可见GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

        # 检查是否有多个GPU
        if ',' in gpu_ids:
            is_parallel = True
            device = torch.device('cuda')
        else:
            device = torch.device('cuda')

    return device, is_parallel

