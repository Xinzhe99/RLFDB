import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(r*r), r*H, r*W],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert ch % (scale_factor * scale_factor) == 0

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.permute(0, 1, 4, 2, 5, 3)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor


def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (r*r)*C, H/r, W/r],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert height % scale_factor == 0
    assert width % scale_factor == 0

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute(0, 1, 3, 5, 2, 4)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'

class MoEDense(nn.Module):
    """混合专家线性层 - 使用硬路由以提高计算效率,并添加负载均衡损失"""

    def __init__(self, in_features, out_features, num_experts=4, top_k=1):
        super(MoEDense, self).__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        # 专家模块保持不变
        self.experts = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for _ in range(self.num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(in_features, self.num_experts),
            nn.Softmax(dim=-1)
        )
        self.k = top_k

    def forward(self, x):
        batch_size, height, width, channels = x.shape

        # 1. 池化操作保持不变
        x_pooled = x.permute(0, 3, 1, 2)
        x_pooled = F.adaptive_avg_pool2d(x_pooled, (1, 1))
        x_pooled = x_pooled.squeeze(-1).squeeze(-1)

        # 2. 计算门控网络输出
        gate_logits = self.gate(x_pooled)  # [B, num_experts]
        _, expert_indices = torch.max(gate_logits, dim=-1)

        # 3. 计算负载均衡损失
        # 计算每个专家的使用频率
        # 3. 计算负载均衡损失时添加平滑处理
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_usage[i] = (expert_indices == i).float().mean()

        # 添加平滑处理，避免出现0值
        epsilon = 1e-6
        expert_usage = expert_usage + epsilon
        expert_usage = expert_usage / expert_usage.sum()  # 重新归一化

        # 理想分布也需要相应调整
        ideal_usage = torch.ones_like(expert_usage) / self.num_experts

        # 使用更稳定的KL散度计算方式
        load_balancing_loss = torch.sum(expert_usage * (torch.log(expert_usage) - torch.log(ideal_usage)))

        # 4. 处理专家输出(保持不变)
        expert_outputs = torch.zeros(batch_size, height, width, self.out_features, device=x.device)
        for expert_idx in range(self.num_experts):
            batch_indices = (expert_indices == expert_idx).nonzero().squeeze(-1)
            if len(batch_indices) > 0:
                expert_input = x[batch_indices]
                expert_output = self.experts[expert_idx](expert_input)
                expert_outputs[batch_indices] = expert_output
        return expert_outputs, load_balancing_loss

class DetectorHead(torch.nn.Module):
    def __init__(self, input_channel, cell_size):
        super(DetectorHead, self).__init__()
        self.cell_size = cell_size
        ##
        self.act = torch.nn.ReLU(inplace=True)
        self.dense = MoEDense(input_channel,pow(cell_size, 2)+1,num_experts=4,top_k=1)
        self.norm = torch.nn.BatchNorm2d(pow(cell_size, 2)+1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)
        x,load_balancing_loss = self.dense(x)
        x = x.permute(0, 3, 1, 2)
        out = self.norm(x)
        prob = self.softmax(out)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = pixel_shuffle(prob, self.cell_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]
        return {'logits':out, 'prob':prob,'load_balancing_loss':load_balancing_loss}

class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4):
        super().__init__()
        hidden_features = min(2048, int(mlp_ratio * dim))
        self.fc1 = nn.Linear(dim, hidden_features, bias=False)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, use_act=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups,
                              padding=kernel_size // 2, bias=False)
        # self.conv =FDConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
        #                       padding=kernel_size // 2, bias=False)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = nn.GELU() if use_act else nn.Identity()

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        return out


class LearnablePool2d(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(1, 1, kernel_size, kernel_size), requires_grad=True)
        nn.init.normal_(self.weight, 0, 0.01)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        weight = self.weight.repeat(self.dim, 1, 1, 1)
        out = nn.functional.conv2d(x, weight, None, self.stride, self.padding, groups=self.dim)
        return self.norm(out)


class ChannelLearnablePool2d(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, groups=dim, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.conv(x)
        return self.norm(out)


class PyramidFC(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, use_dw=False):
        super(PyramidFC, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d

        self.branch_1 = nn.Sequential(
            block(inplanes, kernel_size=3, stride=1, padding=1),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_2 = nn.Sequential(
            block(inplanes, kernel_size=5, stride=2, padding=2),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_3 = nn.Sequential(
            block(inplanes, kernel_size=7, stride=3, padding=3),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.branch_1(x)
        x2 = nn.functional.interpolate(self.branch_2(x), size=(h, w), scale_factor=None, mode='nearest')
        x3 = nn.functional.interpolate(self.branch_3(x), size=(h, w), scale_factor=None, mode='nearest')
        x4 = self.branch_4(x)
        out = self.act(x1 + x2 + x3 + x4)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path=0.0):
        super(BottleNeck, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d
        expand_planes = int(in_planes * expand_ratio)
        mid_planes = int(out_planes * mlp_ratio)

        self.smlp = nn.Sequential(
            PyramidFC(in_planes, expand_planes, kernel_size=3, stride=stride, use_dw=use_dw),
            ConvX(expand_planes, in_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )
        self.cmlp = nn.Sequential(
            ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, use_act=True),
            block(mid_planes, kernel_size=3, stride=stride, padding=1) if stride == 1 else ConvX(mid_planes, mid_planes,
                                                                                                 groups=mid_planes,
                                                                                                 kernel_size=3,
                                                                                                 stride=2,
                                                                                                 use_act=False),
            ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )

        self.skip = nn.Identity()
        if stride == 2 and in_planes != out_planes:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=2, use_act=False),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
            )
        elif stride == 1:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=1, use_act=False),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.smlp(x)) + x
        x = self.drop_path(self.cmlp(x)) + self.skip(x)
        return x

class PFMLP_layer(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path=0.0):
        super(PFMLP_layer, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d
        expand_planes = int(in_planes * expand_ratio)
        mid_planes = int(out_planes * mlp_ratio)

        self.smlp = nn.Sequential(
            PyramidFC(in_planes, expand_planes, kernel_size=3, stride=stride, use_dw=use_dw),
            ConvX(expand_planes, in_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )
        self.cmlp = nn.Sequential(
            ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, use_act=True),
            block(mid_planes, kernel_size=3, stride=stride, padding=1) if stride == 1 else ConvX(mid_planes, mid_planes,
                                                                                                 groups=mid_planes,
                                                                                                 kernel_size=3,
                                                                                                 stride=2,
                                                                                                 use_act=False),
            ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )

        if stride == 2 and in_planes != out_planes:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=2, use_act=False),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
            )
        elif stride == 1:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=1, use_act=False),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x =x.permute(0,2,3,1)
        x = self.drop_path(self.smlp(x))
        x = self.drop_path(self.cmlp(x))
        return x.permute(0,3,1,2)

class RLFDB(nn.Module):
    def __init__(self, dims, layers, block=BottleNeck, expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path_rate=0.,
                 cell_size=8):
        super(RLFDB, self).__init__()
        self.block = block
        self.expand_ratio = expand_ratio
        self.mlp_ratio = mlp_ratio
        self.use_dw = use_dw
        self.drop_path_rate = drop_path_rate

        if isinstance(dims, int):
            dims = [dims // 2, dims, dims * 2, dims * 4, dims * 8]
        else:
            dims = [dims[0] // 2] + dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        # self.first_conv = ConvX(3, dims[0], 1, 3, 2, use_act=True)
        # 1. 修改first_conv层，将stride改为1
        self.first_conv = ConvX(3, dims[0], 1, 3, stride=1, use_act=True)  # xxz把stride=2改为stride=1

        # 2. 修改layer1-4的创建方法，将stride参数改为1
        # 在PFMLP类中修改_make_layers方法的调用：
        self.layer1 = self._make_layers(dims[0], dims[1], layers[0], stride=2, drop_path=dpr[:layers[0]])
        self.layer2 = self._make_layers(dims[1], dims[2], layers[1], stride=2, drop_path=dpr[layers[0]:sum(layers[:2])])
        self.layer3 = self._make_layers(dims[2], dims[3], layers[2], stride=2,
                                        drop_path=dpr[sum(layers[:2]):sum(layers[:3])])
        self.layer4 = self._make_layers(dims[3], dims[4], layers[3], stride=1,
                                        drop_path=dpr[sum(layers[:3]):sum(layers[:4])])

        self.detector_head=DetectorHead(input_channel=dims[4], cell_size=cell_size)


        self.init_params(self)

    def _make_layers(self, inputs, outputs, num_block, stride, drop_path):
        layers = [self.block(inputs, outputs, stride, self.expand_ratio, self.mlp_ratio, self.use_dw, drop_path[0])]

        for i in range(1, num_block):
            layers.append(self.block(outputs, outputs, 1, self.expand_ratio, self.mlp_ratio, self.use_dw, drop_path[i]))

        return nn.Sequential(*layers)

    def init_params(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                try:
                    if 'first' in name:
                        nn.init.normal_(m.weight, 0, 0.01)
                    else:
                        nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                except Exception as e:
                    print(e)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)
        # feat_map = self.att(feat_map)
        outputs = self.detector_head(feat_map)
        return outputs



if __name__ == "__main__":
    input_channels = 3
    height = 256
    width = 256
    # Create a sample input tensor
    batch_size = 1
    device = 'cuda:0' # Make sure to select the correct GPU, changed from 1 to 0.
    x1 = torch.randn(batch_size, input_channels, height, width).to(device)
    model = RLFDB(dims=[32, 64, 128, 256], layers=[2, 2, 6, 2], expand_ratio=3.0,
                     mlp_ratio=3.0, use_dw=True, drop_path_rate=0.05).to(device)
    model.eval()
    t1=time.time()
    out = model(x1)
    print(time.time()-t1)
    logits = out['logits']
    prob = out['prob']

    print(f"Input shape: {x1.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"prob shape: {logits.shape}")

    # Calculate the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f} M")

    # Calculate FLOPs using thop
    flops, params = profile(model, inputs=(x1,))
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Parameters: {params / 1e6:.2f} M")