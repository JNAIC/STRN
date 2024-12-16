import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.spd.nn import BiMap, LogEig, SPDCov2d, DiagonalizingLayer, ReEig


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC_1(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC_1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A3=None, A6=None, spd_A=None, alpha=1, beta=1, gamma=1):

        A6_ex = A6.view(1, 6, 25, 25).expand(self.out_channels // 6, -1, -1, -1)
        A6_ex = A6_ex.contiguous().view(self.out_channels, 25, 25)

        # (4, 3, 64, 25)
        x1, x2 = self.conv1(x).mean(-2), self.conv2(x).mean(-2)
        # (4, 8, 25) (4, 8, 25) (4, 64, 64, 25)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # (4, 64, 25, 25)
        x1 = self.conv4(x1) * alpha + (A3.unsqueeze(0).unsqueeze(0) if A3 is not None else 0) + A6_ex.unsqueeze(
            0) * beta + (spd_A * gamma if spd_A is not None else 0)  # N,C,V,V
        # (4, 64, 25, 25) (1, 1, 25, 25)
        return x1


class CTRGC_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CTRGC_2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, x1):
        # (4, 3, 64, 25)
        x3 = self.conv3(x)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        # (4, 64, 64, 25)
        return x1


# class CTRGC(nn.Module):
#     def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
#         super(CTRGC, self).__init__()
#         self.ctrgc1 = CTRGC_1(in_channels, out_channels, rel_reduction, mid_reduction)
#         self.ctrgc2 = CTRGC_2()
#
#     def forward(self, x):
#         x1 = self.ctrgc1(x)
#         x1 = self.ctrgc2(x, x1)
#         return x1

class unit_spd(nn.Module):
    def __init__(self):
        super(unit_spd, self).__init__()
        self.bimap1 = BiMap(1, 1, 289, 100)
        self.bimap2 = BiMap(1, 1, 100, 25)
        self.logeig = LogEig()

    def forward(self, x):
        x = torch.sinh(self.bimap1(x))
        x = torch.sinh(self.bimap2(x))
        x = self.spd_net(x)

        return x


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A3, A6, coff_embedding=4, residual=True, alpha=None, beta=None,
                 gamma=None):
        super(unit_gcn, self).__init__()
        # (4, 3, 64, 25)
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A3.shape[0]
        self.convs1 = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs1.append(CTRGC_1(in_channels, out_channels))

        self.convs2 = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs2.append(CTRGC_2(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        self.A3 = nn.Parameter(torch.from_numpy(A3.astype(np.float32)))
        self.A6 = nn.Parameter(torch.from_numpy(A6.astype(np.float32)))
        self.alpha = alpha if alpha is not None else nn.Parameter(torch.zeros(1))
        self.beta = beta if beta is not None else nn.Parameter(torch.zeros(1))
        self.gamma = gamma if gamma is not None else nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x, spd_A=None):
        # (4, 3, 64, 25)
        y = None
        A3 = self.A3.cuda(x.get_device())
        A6 = self.A6.cuda(x.get_device())
        # A_se = self.A_SE.cuda(x.get_device())

        A_at = []
        # 直接相加？
        for i in range(self.num_subset):
            A_at.append(self.convs1[i](x, A3[i], A6, spd_A, self.alpha, self.beta, self.gamma))

        A_at = torch.stack(A_at, 0)

        for i in range(self.num_subset):
            z = self.convs2[i](x, A_at[i])
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y, A3.detach(), A6.detach(), A_at[:, 0, 0, :, :].squeeze(1).squeeze(
            1).detach(), self.alpha, self.beta, self.gamma


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A3, A6, stride=1, residual=True, kernel_size=5,
                 dilations=[1, 2], alpha=None, beta=None, gamma=None):
        super(TCN_GCN_unit, self).__init__()
        # (4, 3, 64, 25) , out=64
        self.gcn1 = unit_gcn(in_channels, out_channels, A3, A6, alpha=alpha, beta=beta, gamma=gamma)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, spd_A=None):
        # (4, 3, 64, 25)
        y, A3, A6, A_fn, alpha, beta, gamma = self.gcn1(x, spd_A)
        y = self.relu(self.tcn1(y) + self.residual(x))
        return y, A3, A6, A_fn, alpha, beta, gamma


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        # 维度必须能被num_head 整除
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        # 定义线性变换矩阵
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, alpha=None, beta=None, gamma=None):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.spdmodel = SPDModel(self.graph)
        self.gcn_tcn = GCN_TCN(num_class, num_point, num_person, self.graph, in_channels, drop_out, alpha=alpha,
                               beta=beta, gamma=gamma)

    def forward(self, x):
        device = x.device
        spd_A = self.spdmodel(device)  # 60, 25, 25
        x = self.gcn_tcn(x, spd_A)
        return x


class SPDModel(nn.Module):
    def __init__(self, graph):
        super(SPDModel, self).__init__()
        self.spd_A = Variable(torch.from_numpy(graph.spd_A.astype(np.float32)), requires_grad=True)
        if self.spd_A.shape[1] == 25:
            self.module = nn.Sequential(
                SPDCov2d(1, 6, kernel_size=5, stride=2),
                SPDCov2d(6, 6, kernel_size=5, stride=2),
                ReEig(),
                SPDCov2d(6, 6, kernel_size=5, stride=2),
                SPDCov2d(6, 6, kernel_size=5, stride=3),
                ReEig(),
                LogEig()
            )
        elif self.spd_A.shape[1] == 17:
            self.module = nn.Sequential(
                SPDCov2d(1, 6, kernel_size=5, stride=2),
                SPDCov2d(6, 6, kernel_size=5, stride=2),
                ReEig(),
                SPDCov2d(6, 6, kernel_size=5, stride=2),
                SPDCov2d(6, 6, kernel_size=5, stride=3),
                ReEig(),
                LogEig()
            )

    def forward(self, device):
        K, N, N = self.spd_A.shape
        x = self.spd_A.view(6, -1).to(device)

        x = torch.cov(x.T)
        x = x + 0.001 * torch.eye(N * N, device=device)
        x = x.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 625, 625]

        x = self.module(x)

        return x


class GCN_TCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, alpha=None, beta=None, gamma=None):
        super(GCN_TCN, self).__init__()

        A3 = graph.A3  # 3,25,25
        A6 = graph.A6  # 6,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 60
        # (4, 3, 64, 25)
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A3, A6, residual=False, alpha=alpha, beta=beta, gamma=gamma)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A3, A6, alpha=alpha, beta=beta, gamma=gamma)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A3, A6, alpha=alpha, beta=beta, gamma=gamma)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A3, A6, alpha=alpha, beta=beta, gamma=gamma)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A3, A6, stride=2, alpha=alpha, beta=beta, gamma=gamma)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A3, A6, alpha=alpha, beta=beta, gamma=gamma)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A3, A6, alpha=alpha, beta=beta, gamma=gamma)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A3, A6, stride=2, alpha=alpha, beta=beta,
                               gamma=gamma)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A3, A6, alpha=alpha, beta=beta, gamma=gamma)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A3, A6, alpha=alpha, beta=beta, gamma=gamma)

        self.up = nn.Conv2d(6, base_channel, kernel_size=1)
        self.up1 = nn.Conv2d(base_channel, base_channel * 2, kernel_size=1)
        self.up2 = nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=1)

        # Retrospect Model
        self.first_tram = nn.Sequential(
            nn.AvgPool2d((4, 1)),
            nn.Conv2d(60, 240, 1),
            nn.BatchNorm2d(240),
            nn.ReLU()
        )
        self.second_tram = nn.Sequential(
            nn.AvgPool2d((2, 1)),
            nn.Conv2d(120, 240, 1),
            nn.BatchNorm2d(240),
            nn.ReLU()
        )

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        self.aalpha = alpha
        self.abeta = beta
        self.agamma = gamma

    def forward(self, x, spd_A=None):
        log = {'image': {}, 'pram': {}}
        """
            N 视频个数(batch_size)
            C = 3 (X,Y,S)代表一个点的信息(位置+预测的可能性)
            T = 64 一个视频的帧数paper规定是64帧，不足的重头循环，多的clip
            V 25 数据集中25个结点
            M = 2 人数，paper中将人数限定在最大2个人
        """
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # N C T V M -> N M V C T -> N MVC T
        # print(x.shape)
        x = self.data_bn(x)  # batch_normalize
        # print(x.shape)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # N MVC T -> N M V C T -> N M C T V -> NM C T V

        spd_A = self.up(spd_A)  # 1, 60, 25, 25
        x = self.l1(x)[0]  # (N*M, 60 , 64, 25)
        x = self.l2(x)[0]  # (N*M, 60, 64, 25)
        x = self.l3(x)[0]  # (N*M, 60, 64, 25)
        x = self.l4(x, spd_A=spd_A)[0]  # (N*M, 60, 64, 25)
        x2 = x
        spd_A = self.up1(spd_A)
        x = self.l5(x)[0]  # (N*M, 120, 32, 25)
        x = self.l6(x)[0]  # (N*M, 120, 32, 25)
        x = self.l7(x, spd_A=spd_A)[0]  # (N*M, 120, 32, 25)
        x3 = x
        spd_A = self.up2(spd_A)
        x = self.l8(x)[0]  # (N*M, 240, 16, 25)
        x = self.l9(x)[0]  # (N*M, 240, 16, 25)
        x, A3, A6, A_fn, alpha, beta, gamma = self.l10(x, spd_A=spd_A)  # (N*M, 240, 16, 25)

        x2 = self.first_tram(x2)
        x3 = self.second_tram(x3)
        x = x + x2 + x3

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)  # N, M, C, T*V
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        log['image'] = {
            'A3': A3,
            'A6': A6,
            'A_fn': A_fn,
            'A_spd': spd_A.detach()
        }

        if not self.aalpha:
            log['pram']['alpha'] = alpha.detach()
        if not self.abeta:
            log['pram']['beta'] = beta.detach()
        if not self.aalpha:
            log['pram']['agamma'] = gamma.detach()

        return self.fc(x), log
