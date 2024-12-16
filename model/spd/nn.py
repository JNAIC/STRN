import torch
import torch as th
import torch.nn as nn
# from torch.autograd import Function as F
from . import functional
import geoopt
from geoopt import SymmetricPositiveDefinite
from torch.nn import functional as F
import geoopt.manifolds.symmetric_positive_definite
# import SymmetricPositiveDefinite
from geoopt.manifolds import Stiefel

dtype = th.float32
device = th.device('cpu')


class SPDCov2d(nn.Module):
    # 只需要训练V，直接pytorch优化即可
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(SPDCov2d, self).__init__()
        self.V = nn.Parameter(torch.randn((out_channel, in_channel, kernel_size, kernel_size),
                                          dtype=torch.float32), requires_grad=True)
        # W = th.zeros((out_channel, in_channel, kernel_size, kernel_size), dtype=th.float32) +\
        #     th.eye(kernel_size, dtype=th.float32)
        # for i in range(kernel_size):
        #    W[:, :, i, i] += (i + 1) * 1e-6
        # self.W = geoopt.ManifoldParameter(W, manifold=SymmetricPositiveDefinite())
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

    def forward(self, x):
        V = self.V
        V_T = torch.einsum('oiuv->oivu', V)
        W = torch.einsum('oiuv, oivw -> oiuw', V_T, V) + 1e-10 * torch.eye(self.kernel_size, device=V.device)  # 构建SPD
        x = F.conv2d(x, W, stride=self.stride, padding=self.padding)
        return x


class DiagonalizingLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DiagonalizingLayer, self).__init__()
        self.reeig = ReEig()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if in_dim != out_dim:
            self.fn = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.reeig(x).view(x.shape[1], -1, 1)
        if self.in_dim != self.out_dim:
            x = self.fn(x)
        x = x.view(x.shape[0], -1)
        x = torch.stack([torch.diag(xi) for xi in x])

        return x


def bimap(X, W):
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    return W.t().matmul(X.to(torch.float32)).matmul(W)


class BiMapGeo(nn.Module):
    def __init__(self, ho, hi, ni, no):
        super(BiMapGeo, self).__init__()
        self.W = geoopt.ManifoldParameter(x=th.empty(ho, hi, ni, no), manifold=Stiefel)
        self._ho = ho
        self._hi = hi
        self._ni = ni
        self._no = no
        functional.init_bimap_parameter(self.W)

    def forward(self, x):
        W = self.W
        batch_size, channels_in, n_in, _ = x.shape
        channels_out, _, _, n_out = W.shape
        P = th.zeros(batch_size, channels_out, n_out, n_out, dtype=x.dtype, device=x.device)
        for co in range(channels_out):
            P[:, co, :, :] = sum([bimap(x[:, ci, :, :], W[co, ci, :, :]) for ci in range(channels_in)])
        return P


class BiMap(nn.Module):
    """
    Input X: (batch_size,hi) SPD matrices of size (ni,ni)
    Output P: (batch_size,ho) of bilinearly mapped matrices of size (no,no)
    Stiefel parameter of size (ho,hi,ni,no)
    """

    def __init__(self, ho, hi, ni, no):
        super(BiMap, self).__init__()
        self._W = functional.StiefelParameter(th.empty(ho, hi, ni, no, dtype=dtype, device=device))
        self._ho = ho;
        self._hi = hi;
        self._ni = ni;
        self._no = no
        functional.init_bimap_parameter(self._W)

    def forward(self, X):
        return functional.bimap_channels(X, self._W)


class LogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        device = P.device
        return functional.LogEig.apply(P.to('cpu')).to(device)


class SqmEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of sqrt eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.SqmEig.apply(P)


class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        device = P.device
        return functional.ReEig.apply(P.to('cpu')).to(device)


class BaryGeom(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''

    def forward(self, x):
        return functional.BaryGeom(x)


class BatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized matrices
    SPD parameter of size (n,n)
    """

    def __init__(self, n):
        super(__class__, self).__init__()
        self.momentum = 0.1
        self.running_mean = th.eye(n, dtype=dtype)  ################################
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        self.weight = functional.SPDParameter(th.eye(n, dtype=dtype))

    def forward(self, X):
        N, h, n, n = X.shape
        X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, n, N * h, 1).permute(2, 3, 0, 1).contiguous()
        if (self.training):
            mean = functional.BaryGeom(X_batched)
            with th.no_grad():
                self.running_mean.data = functional.geodesic(self.running_mean, mean, self.momentum)
            X_centered = functional.CongrG(X_batched, mean, 'neg')
        else:
            X_centered = functional.CongrG(X_batched, self.running_mean, 'neg')
        X_normalized = functional.CongrG(X_centered, self.weight, 'pos')
        return X_normalized.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()


class CovPool(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """

    def __init__(self, reg_mode='mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        return functional.cov_pool(f, self._reg_mode)


class CovPoolBlock(nn.Module):
    """
    Input f: L blocks of temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,L,n,T)
    Output X: L covariance matrices, shape (batch_size,L,1,n,n)
    """

    def __init__(self, reg_mode='mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        ff = [functional.cov_pool(f[:, i, :, :], self._reg_mode)[:, None, :, :, :] for i in range(f.shape[1])]
        return th.cat(ff, 1)


class CovPoolMean(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """

    def __init__(self, reg_mode='mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        return functional.cov_pool_mu(f, self._reg_mode)
