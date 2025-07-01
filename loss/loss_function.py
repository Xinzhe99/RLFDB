import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import tensor_op

def detector_loss(keypoint_map, logits, valid_mask=None, grid_size=8, device='cpu'):
    labels = keypoint_map.float()#to [B, 1, H, W]
    labels = tensor_op.pixel_shuffle_inv(labels, grid_size) # to [B,64,H/8,W/8]
    B,C,h,w = labels.shape#h=H/grid_size,w=W/grid_size
    labels = torch.cat([2*labels, torch.ones([B,1,h,w],device=device)], dim=1)
    # Add a small random matrix to randomly break ties in argmax
    labels = torch.argmax(labels + torch.zeros(labels.shape,device=device).uniform_(0,0.1),dim=1)#B*65*Hc*Wc

    #如果原始 8×8 区域包含关键点（原始值为 1）： 经过 2 * labels，对应的通道在前 64 个通道中的值是 2。第 65 个通道的值是 1。因为 2 > 1（通常情况下，除非随机噪声干扰），argmax 会选择对应关键点的索引（0 到 63 之间）。
    #如果原始 8×8 区域不包含关键点（原始值为 0）： 经过 2 * labels，对应的通道在前 64 个通道中的值是 0。第 65 个通道的值是 1。因为 1 > 0，argmax 会选择索引 64。

    # Mask the pixels if bordering artifacts appear
    valid_mask = torch.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = tensor_op.pixel_shuffle_inv(valid_mask, grid_size)#[B, 64, H/8, W/8]
    valid_mask = torch.prod(valid_mask, dim=1).unsqueeze(dim=1).type(torch.float32)#[B,1,H/8,W/8]

    ## method 1
    ce_loss = F.cross_entropy(logits, labels, reduction='none',)
    valid_mask = valid_mask.squeeze(dim=1)
    loss = torch.divide(torch.sum(ce_loss * valid_mask, dim=(1, 2)), torch.sum(valid_mask + 1e-6, dim=(1, 2)))
    loss = torch.mean(loss)
    return loss
