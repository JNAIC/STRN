import torch.nn as nn
import torch
from torch.nn import Parameter
from utils import functional, manifolds


class Gaussian_Embedding(nn.Module):  # 高斯嵌入模块（对应GeomNet中公式 20 21 22 23 24）
    def __init__(self, k):
        super(Gaussian_Embedding, self).__init__()
        self.k = k

    def forward(self, x):

        N, C, W, T, V, M = x.shape
        x = x.permute(0, 3, 4, 5, 2, 1).contiguous().view(-1, C, W)
        x_mean = x.mean(2, keepdim=True)  # 计算每个 token 的平均值
        x_center = x - x_mean
        x_cov = (x_center.matmul(x_center.transpose(-1, -2))) / (W - 1)  # 计算协方差
        tr = torch.sum(torch.diagonal(x_cov, dim1=-2, dim2=-1), dim=-1).unsqueeze(-1).unsqueeze(-1)  # calculate trace
        x_cov = x_cov / tr  # 迹归一化
        # x_cov = x_cov + tr * torch.eye(C, device='cuda') * 1e-3  # 添加扰动
        x_cov = x_cov + tr * torch.eye(C) * 1e-3  # 添加扰动
        # ？迹归一化了，还需要*tr吗？

        x_mean = x_mean.contiguous().view(N, T, V, M, C, -1)
        x_cov = x_cov.contiguous().view(N, T, V, M, C, C)

        det_sigma = torch.det(x_cov).unsqueeze(-1).unsqueeze(-1)  # 计算协方差矩阵的行列式
        power = -1.0 / (C + self.k)
        det_sigma_power = det_sigma.pow(power)
        if self.k == 0:
            return det_sigma_power * x_cov

        # 创建单位矩阵 I_k，它的形状是 (k, k)
        I_k = torch.eye(self.k).repeat(N, T, V, M, 1, 1)  # 这将创建形状为 (N, T,V, M, k, k) 的单位矩阵
        # 计算 Σ + k * μ * μ^T
        mu_outer_product = x_mean.matmul(x_mean.transpose(-1, -2))
        sigma_plus_mu_outer = x_cov + self.k * mu_outer_product
        # 创建扩展后的块矩阵
        extended_block_matrix = torch.zeros((N, T, V, M, C + self.k, C + self.k))
        # 填充块矩阵的四个部分
        extended_block_matrix[:, :, :, :, :C, :C] = sigma_plus_mu_outer
        extended_block_matrix[:, :, :, :, :C, -self.k:] = x_mean
        extended_block_matrix[:, :, :, :, -self.k:, :C] = x_mean.transpose(-1, -2)
        extended_block_matrix[:, :, :, :, -self.k:, -self.k:] = I_k

        # 将均值和协方差嵌入SPD空间, P 的形状为 (N, T, V, M, C + k, C + k)
        P = det_sigma_power * extended_block_matrix
        return P


def lower_triangle(x):
    C = x.shape[-1]
    # 获取矩阵的下三角部分，不包括对角线（diagonal=1设置为从对角线向上偏移）
    lower_triangle_without_diagonal = torch.tril(x, diagonal=-1)
    # 将对角线外的下三角部分乘以根号2
    scaled_lower_triangle_without_diagonal = 2.0 ** 0.5 * lower_triangle_without_diagonal
    # 提取原始矩阵的对角线元素
    diagonals = torch.diagonal(x, dim1=-2, dim2=-1)
    # 用这些对角线元素创建一个对角线矩阵
    diagonal_matrix = torch.diag_embed(diagonals)
    # 将缩放的下三角矩阵与对角线矩阵相加
    lower_triangle_matrix = scaled_lower_triangle_without_diagonal + diagonal_matrix
    rows, cols = torch.tril_indices(C, C)
    return lower_triangle_matrix[:, :, rows, cols]


class Embedding_SPD(nn.Module):
    def __init__(self, k, input_dims):
        super(Embedding_SPD, self).__init__()
        self.k = k
        self.input_dims = input_dims
        self.dim = self.k + self.input_dims * (self.input_dims + 1) // 2
        self.manifold = manifolds.SymmetricPositiveDefinite()
        self.W = Parameter(torch.eye(self.input_dims))
        self.W_lw = Parameter(torch.eye(self.dim))

    def forward(self, x):
        N, T, V, M, C, _ = x.shape
        x = x.view(N, -1, C, C)
        x_mean = self.manifold.barycenter(x, 1, 1)  # aim下的均值
        W_pt = functional.sym_expm.apply(functional.ensure_sym(self.W))
        x_center = self.manifold.transp_via_identity(self.manifold.logmap(x_mean, x), x_mean, W_pt)  # pt
        # x_center = self.manifold.logmap(x_mean, x)  # aim下的Log
        x_center = lower_triangle(x_center)
        x_cov = (x_center.transpose(-1, -2).matmul(x_center)) / (T * V * M - 1)  # 计算协方差
        C = x_center.shape[-1]
        tr = torch.sum(torch.diagonal(x_cov, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1)  # calculate trace
        x_cov = x_cov / tr  # 迹归一化
        x_cov = x_cov + torch.eye(C, device=x.get_device()) * 1e-5  # 添加扰动
        L = torch.linalg.cholesky(x_cov, upper=False)  # cholesky分解
        if self.k == 0:
            return L @ torch.linalg.cholesky(functional.sym_expm.apply(functional.ensure_sym(self.W_lw)), upper=False)

        # 创建单位矩阵 I_k，它的形状是 (k, k)
        I_k = torch.eye(self.k).repeat(N, 1, 1)  # 这将创建形状为 (N,  k, k) 的单位矩阵
        # 创建扩展后的块矩阵
        extended_block_matrix = torch.zeros((N, C + self.k, C + self.k), device=x.get_device())
        # 填充块矩阵的四个部分
        extended_block_matrix[:, :C, :C] = L
        # extended_block_matrix[:, :C, -self.k:] =  # 右上角
        extended_block_matrix[:, -self.k:, :C] = lower_triangle(functional.sym_logm.apply(x_mean))  # 左下角
        extended_block_matrix[:, -self.k:, -self.k:] = I_k

        return extended_block_matrix @ torch.linalg.cholesky(
            functional.sym_expm.apply(functional.ensure_sym(self.W_lw)), upper=False)


