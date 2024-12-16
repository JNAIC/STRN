'''
#####################################################################################################################
Discription:

The utility functions in this file offer the forward function for the ReEig layer, the LogEig layer, and Riemannian Batch
Normalization in geometric models (Tensor-CSPNet and Graph-CSPNet). Additionally, they provide an optimizer for network
architecture. The primary functions and classes are mainly derived from the following repository:

https://gitlab.lip6.fr/schwander/torchspdnet
https://github.com/adavoudi/spdnet
https://github.com/zhiwu-huang/SPDNet
https://github.com/YirongMao/SPDNet

#######################################################################################################################
'''
import math

import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function as F, gradcheck
import torch.optim
import geoopt
from typing import Callable, Tuple
from typing import Any
from torch.types import Number

# define the epsilon precision depending on the tensor datatype
EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class MixOptimizer():
    """ Optimizer with mixed constraints """

    def __init__(self, parameters, optimizer=torch.optim.SGD, lr=1e-2, *args, **kwargs):
        parameters = list(parameters)
        parameters = [param for param in parameters if param.requires_grad]
        self.lr = lr
        # self.stiefel_parameters = [param for param in parameters if param.__class__.__name__=='StiefelParameter']
        # self.stiefel_block_parameters = [param for param in parameters if param.__class__.__name__=='StiefelBlockParameter']
        # self.spd_parameters = [param for param in parameters if param.__class__.__name__=='SPDParameter']
        # self.weight_vector_parameters = [param for param in parameters if param.__class__.__name__=='WeightVectorParameter']
        self.riemannian_parameters = [param for param in parameters if param.__class__.__name__ == 'StiefelParameter']
        self.other_parameters = [param for param in parameters if param.__class__.__name__ == 'Parameter']

        # self.stiefel_optim = StiefelOptim(self.stiefel_parameters, self.lr)
        # self.stiefel_block_optim = StiefelBlockOptim(self.stiefel_block_parameters, self.lr)
        # self.spd_optim = SPDOptim(self.spd_parameters, self.lr)
        # self.weight_vector_optim = WeightVectorOptim(self.weight_vector_parameters, self.lr)

        self.riemannian_optim = geoopt.optim.RiemannianSGD(self.riemannian_parameters, lr, *args, **kwargs)
        self.optim = optimizer(self.other_parameters, lr, *args, **kwargs)

    def step(self):
        self.optim.step()
        if self.riemannian_optim:
            self.riemannian_optim.step()
        # self.stiefel_optim.step()
        # self.stiefel_block_optim.step()
        # self.spd_optim.step()
        # self.weight_vector_optim.step()

    def zero_grad(self):
        self.optim.zero_grad()
        if self.riemannian_optim:
            self.riemannian_optim.zero_grad()
        # self.stiefel_optim.zero_grad()
        # self.spd_optim.zero_grad()

    def adjust_learning_rate(self, lr):
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        for param_group in self.riemannian_optim.param_groups:
            param_group['lr'] = lr

        self.lr = lr

        # self.stiefel_optim.lr = lr
        # self.stiefel_block_optim.lr = lr
        # self.spd_optim.lr = lr
        # self.weight_vector_optim.lr = lr
        # self.lr = lr


def proj_tanX_stiefel(x, X):
    """ Projection of x in the Stiefel manifold's tangent space at X """
    return x - X.matmul(x.transpose(-2, -1)).matmul(X)


def ExpX_stiefel(x, X):
    """ Exponential mapping of x on the Stiefel manifold at X (retraction operation) """
    a = X + x
    Q = th.zeros_like(a)
    for i in range(a.shape[0]):
        q, _ = gram_schmidt(a[i])
        Q[i] = q
    return Q


def proj_tanX_spd(x, X):
    """ Projection of x in the SPD manifold's tangent space at X """
    return X.matmul(sym(x)).matmul(X)


# V is a in M(n,N); output W an semi-orthonormal free family of Rn; we consider n >= N
# also returns R such that WR is the QR decomposition
def gram_schmidt(V):
    n, N = V.shape  # dimension, cardinal
    W = th.zeros_like(V)
    R = th.zeros((N, N)).double().to(V.device)
    W[:, 0] = V[:, 0] / th.norm(V[:, 0])
    R[0, 0] = W[:, 0].dot(V[:, 0])
    for i in range(1, N):
        proj = th.zeros(n).double().to(V.device)
        for j in range(i):
            proj = proj + V[:, i].dot(W[:, j]) * W[:, j]
            R[j, i] = W[:, j].dot(V[:, i])
        if (isclose(th.norm(V[:, i] - proj), th.DoubleTensor([0]).to(V.device))):
            W[:, i] = V[:, i] / th.norm(V[:, i])
        else:
            W[:, i] = (V[:, i] - proj) / th.norm(V[:, i] - proj)
        R[i, i] = W[:, i].dot(V[:, i])
    return W, R


def isclose(a, b, rtol=1e-05, atol=1e-08):
    return ((a - b).abs() <= (atol + rtol * b.abs())).all()


def sym(X):
    if (len(X.shape) == 2):
        if isinstance(X, np.ndarray):
            return 0.5 * (X + X.T.conj())
        else:
            return 0.5 * (X + X.t())
    elif (len(X.shape) == 3):
        if isinstance(X, np.ndarray):
            return 0.5 * (X + X.transpose([0, 2, 1]))
        else:
            return 0.5 * (X + X.transpose(1, 2))
    elif (len(X.shape) == 4):
        if isinstance(X, np.ndarray):
            return 0.5 * (X + X.transpose([0, 1, 3, 2]))
        else:
            return 0.5 * (X + X.transpose(2, 3))


def ensure_sym(A: Tensor) -> Tensor:
    """Ensures that the last two dimensions of the tensor are symmetric.
    Parameters
    ----------
    A : torch.Tensor
        with the last two dimensions being identical
    -------
    Returns : torch.Tensor
    """
    return 0.5 * (A + A.transpose(-1, -2))


class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass


class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass


def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape  # batch size,channel depth,dimension
    U, S = th.zeros_like(P, device=P.device), th.zeros(batch_size, channels, n, dtype=torch.float32, device=P.device)

    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'eig'):
                # This is for v_pytorch >= 1.9;
                s, U[i, j] = th.linalg.eig(P[i, j][None, :])
                S[i, j] = s[:, 0]
            elif (eig_mode == 'svd'):
                U[i, j], S[i, j], _ = th.svd(P[i, j])

    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    S_fn_deriv = BatchDiag(op.fn_deriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0
    L[L == np.inf] = 0
    L[th.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp


class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Log_op)


class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)


class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Exp_op, eig_mode='eig')
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Exp_op)


class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqm_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqm_op)


class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqminv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqminv_op)


class PowerEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P, power):
        Power_op._power = power
        X, U, S, S_fn = modeig_forward(P, Power_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Power_op), None


def geodesic(A, B, t):
    """
    Geodesic from A to B at step t
    :param A: SPD matrix (n,n) to start from
    :param B: SPD matrix (n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPD matrix (n,n) along the geodesic
    """
    M = CongrG(PowerEig.apply(CongrG(B, A, 'neg'), t), A, 'pos')[0, 0]
    return M


def dist_riemann(x, y):
    """
    Riemannian distance between SPD matrices x and SPD matrix y
    :param x: batch of SPD matrices (batch_size,1,n,n)
    :param y: single SPD matrix (n,n)
    :return:
    """
    return LogEig.apply(CongrG(x, y, 'neg')).view(x.shape[0], x.shape[1], -1).norm(p=2, dim=-1)


def CongrG(P, G, mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if (mode == 'pos'):
        GG = SqmEig.apply(G[None, None, :, :])
    elif (mode == 'neg'):
        GG = SqminvEig.apply(G[None, None, :, :])
    PP = GG.matmul(P).matmul(GG)
    return PP


def LogG(x, X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    return CongrG(LogEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def ExpG(x, X):
    """ Exponential mapping of x on the SPD manifold at X """
    return CongrG(ExpEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size, channels, n = P.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=torch.float32, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i, j] = P[i, j].diag()
    return Q


def karcher_step(x, G, alpha):
    """
    One step in the Karcher flow
    """
    x_log = LogG(x, G)
    G_tan = x_log.mean(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def BaryGeom(x):
    """
    F which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    """
    k = 1
    alpha = 1
    with th.no_grad():
        G = th.mean(x, dim=0)[0, :, :]
        for _ in range(k):
            G = karcher_step(x, G, alpha)
        return G


class Log_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.log(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 1 / S


class Re_op():
    """ Log function and its derivative """
    _threshold = 1e-4

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class Sqm_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 0.5 / th.sqrt(S)


class Sqminv_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return 1 / th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return -0.5 / th.sqrt(S) ** 3


class Power_op():
    """ Power function and its derivative """
    _power = 1

    @classmethod
    def fn(cls, S, param=None):
        return S ** cls._power

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (cls._power) * S ** (cls._power - 1)


class Exp_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.exp(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return th.exp(S)


class sym_modeig:
    """Basic class that modifies the eigenvalues with an arbitrary elementwise function
    """

    @staticmethod
    def forward(M: Tensor, fun: Callable[[Tensor], Tensor], fun_param: Tensor = None,
                ensure_symmetric: bool = False, ensure_psd: bool = False, decom_mode='eigh') -> Tensor:
        """Modifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        M : torch.Tensor
            (batch) of symmetric matrices
        fun : Callable[[Tensor], Tensor]
            elementwise function
        ensure_symmetric : bool = False (optional)
            if ensure_symmetric=True, then M is symmetrized
        ensure_psd : bool = False (optional)
            if ensure_psd=True, then the eigenvalues are clamped so that they are > 0
        -------
        Returns : torch.Tensor with modified eigenvalues
        """
        if ensure_symmetric:
            M = ensure_sym(M)

        # compute the eigenvalues and vectors
        if decom_mode == 'eigh':
            s, U = torch.linalg.eigh(M)
        elif decom_mode == 'svd':
            U, s, _ = torch.linalg.svd(M)
        if ensure_psd:
            s = s.clamp(min=EPS[s.dtype])

        # modify the eigenvalues
        smod = fun(s, fun_param)
        X = U @ torch.diag_embed(smod) @ U.transpose(-1, -2)

        return X, s, smod, U

    @staticmethod
    def backward(dX: Tensor, s: Tensor, smod: Tensor, U: Tensor,
                 fun_der: Callable[[Tensor], Tensor], fun_der_param: Tensor = None) -> Tensor:
        """
        Backpropagates the derivatives

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        dX : torch.Tensor
            (batch) derivatives that should be backpropagated
        s : torch.Tensor
            eigenvalues of the original input
        smod : torch.Tensor
            modified eigenvalues
        U : torch.Tensor
            eigenvector of the input
        fun_der : Callable[[Tensor], Tensor]
            elementwise function derivative
        -------
        Returns : torch.Tensor containing the backpropagated derivatives
        """

        # compute Lowener matrix
        # denominator
        L_den = s[..., None] - s[..., None].transpose(-1, -2)
        # find cases (similar or different eigenvalues, via threshold)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        # case: sigma_i != sigma_j
        L_num_ne = smod[..., None] - smod[..., None].transpose(-1, -2)
        L_num_ne[is_eq] = 0
        # case: sigma_i == sigma_j
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[..., None] + sder[..., None].transpose(-1, -2))
        L_num_eq[~is_eq] = 0
        # compose Loewner matrix
        L = (L_num_ne + L_num_eq) / L_den
        dM = U @ (L * (U.transpose(-1, -2) @ ensure_sym(dX) @ U)) @ U.transpose(-1, -2)
        return dM


class sym_logm(F):
    """
    Computes the matrix logarithm for a batch of SPD matrices.
    Ensures that the input matrices are SPD by clamping eigenvalues.
    During backprop, the update along the clamped eigenvalues is zeroed
    """

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        # ensure that the eigenvalues are positive
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        # compute derivative
        sder = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


def spd_mean_kracher_flow(X: Tensor, G0: Tensor = None, maxiter: int = 50, dim=0, weights=None, return_dist=False,
                          return_XT=False) -> Tensor:
    if X.shape[dim] == 1:
        if return_dist:
            return X, torch.tensor([0.0], dtype=X.dtype, device=X.device)
        else:
            return X

    if weights is None:
        n = X.shape[dim]
        weights = torch.ones((*X.shape[:-2], 1, 1), dtype=X.dtype, device=X.device)
        weights /= n

    if G0 is None:
        G = (X * weights).sum(dim=dim, keepdim=True)
        # G = torch.enisum('ntvmqp,vj->ntjmqp',X, weights)
    else:
        G = G0.clone()

    nu = 1.
    dist = tau = crit = torch.finfo(X.dtype).max
    i = 0

    while (crit > EPS[X.dtype]) and (i < maxiter) and (nu > EPS[X.dtype]):
        i += 1

        Gsq, Ginvsq = sym_invsqrtm2.apply(G)
        XT = sym_logm.apply(Ginvsq @ X @ Ginvsq)
        GT = (XT * weights).sum(dim=dim, keepdim=True)
        # GT = torch.enisum('ntvmqp,vj->ntjmqp', XT, weights)
        G = Gsq @ sym_expm.apply(nu * GT) @ Gsq

        if return_dist:
            dist = torch.norm(XT - GT, p='fro', dim=(-2, -1))
        crit = torch.norm(GT, p='fro', dim=(-2, -1)).max()
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    if return_dist:
        return G, dist
    if return_XT:
        return G, XT
    return G


def broadcast_dims(A: torch.Size, B: torch.Size, raise_error: bool = True) -> Tuple:
    """Return the dimensions that can be broadcasted.
    Parameters
    ----------
    A : torch.Size
        shape of first tensor
    B : torch.Size
        shape of second tensor
    raise_error : bool (=True)
        flag that indicates if an error should be raised if A and B cannot be broadcasted
    -------
    Returns : torch.Tensor
    """
    # check if the tensors can be broadcasted
    if raise_error:
        if len(A) != len(B):
            raise ValueError('The number of dimensions must be equal!')

    tdim = torch.tensor((A, B), dtype=torch.int32)

    # find differing dimensions
    bdims = tuple(torch.where(tdim[0].ne(tdim[1]))[0].tolist())

    # check if one of the different dimensions has size 1
    if raise_error:
        if not tdim[:, bdims].eq(1).any(dim=0).all():
            raise ValueError('Broadcast not possible! One of the dimensions must be 1.')

    return bdims


def sum_bcastdims(A: Tensor, shape_out: torch.Size) -> Tensor:
    """Returns a tensor whose values along the broadcast dimensions are summed.
    Parameters
    ----------
    A : torch.Tensor
        tensor that should be modified
    shape_out : torch.Size
        desired shape of the tensor after aggregation
    -------
    Returns : the aggregated tensor with the desired shape
    """
    bdims = broadcast_dims(A.shape, shape_out)

    if len(bdims) == 0:
        return A
    else:
        return A.sum(dim=bdims, keepdim=True)


def randn_sym(shape, **kwargs):
    ndim = shape[-1]
    X = torch.randn(shape, **kwargs)
    ixs = torch.tril_indices(ndim, ndim, offset=-1)
    X[..., ixs[0], ixs[1]] /= math.sqrt(2)
    X[..., ixs[1], ixs[0]] = X[..., ixs[0], ixs[1]]
    return X


def spd_2point_interpolation(A: Tensor, B: Tensor, t: Number) -> Tensor:
    """
    A with 1-t, B with t
    """
    rm_sq, rm_invsq = sym_invsqrtm2.apply(A)
    return rm_sq @ sym_powm.apply(rm_invsq @ B @ rm_invsq, torch.tensor(t)) @ rm_sq


class reverse_gradient(F):
    """
    Reversal of the gradient
    Parameters
    ---------
    scaling : Number
        A constant number that is multiplied to the sign-reversed gradients (1.0 default)
    """

    @staticmethod
    def forward(ctx, x, scaling=1.0):
        ctx.scaling = scaling
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.scaling
        return grad_output, None


class sym_reeig(F):
    """
    Rectifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).
    """

    @staticmethod
    def value(s: Tensor, threshold: Tensor) -> Tensor:
        return s.clamp(min=threshold.item())

    @staticmethod
    def derivative(s: Tensor, threshold: Tensor) -> Tensor:
        return (s > threshold.item()).type(s.dtype)

    @staticmethod
    def forward(ctx: Any, M: Tensor, threshold: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_reeig.value, threshold, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, threshold)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, threshold = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_reeig.derivative, threshold), None, None

    @staticmethod
    def tests():
        """
        Basic unit tests and test to check gradients
        """
        ndim = 2
        nb = 1
        # generate random base SPD matrix
        A = torch.randn((1, ndim, ndim), dtype=torch.double)
        U, s, _ = torch.linalg.svd(A)

        threshold = torch.tensor([1e-3], dtype=torch.double)

        # generate batches
        # linear case (all eigenvalues are above the threshold)
        s = threshold * 1e1 + torch.rand((nb, ndim), dtype=torch.double) * threshold
        M = U @ torch.diag_embed(s) @ U.transpose(-1, -2)

        assert (sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert (gradcheck(sym_reeig.apply, (M, threshold, True)))

        # non-linear case (some eigenvalues are below the threshold)
        s = torch.rand((nb, ndim), dtype=torch.double) * threshold
        s[::2] += threshold
        M = U @ torch.diag_embed(s) @ U.transpose(-1, -2)
        assert (~sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert (gradcheck(sym_reeig.apply, (M, threshold, True)))

        # linear case, all eigenvalues are identical
        s = torch.ones((nb, ndim), dtype=torch.double)
        M = U @ torch.diag_embed(s) @ U.transpose(-1, -2)
        assert (sym_reeig.apply(M, threshold, True).allclose(M))
        M.requires_grad_(True)
        assert (gradcheck(sym_reeig.apply, (M, threshold, True)))


class sym_abseig(F):
    """
    Computes the absolute values of all eigenvalues for a batch symmetric matrices.
    """

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.abs()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        return s.sign()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_abseig.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_abseig.derivative), None


class sym_expm(F):
    """
    Computes the matrix exponential for a batch of symmetric matrices.
    """

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def clip_gradients(gradients: Tensor, max_norm: float, norm_type: float) -> Tensor:
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(gradients, max_norm, norm_type=norm_type)
        return gradients

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm.value, ensure_symmetric=ensure_symmetric, decom_mode='eigh')
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        # grad = sym_modeig.backward(dX, s, smod, U, sym_expm.derivative)
        # min_norm = min(3, 5 / grad.norm(2))
        # grad = sym_expm.clip_gradients(grad, min_norm, norm_type=2)
        return sym_modeig.backward(dX, s, smod, U, sym_expm.derivative), None


class sym_powm(F):
    """
    Computes the matrix power for a batch of symmetric matrices.
    """

    @staticmethod
    def value(s: Tensor, exponent: Tensor) -> Tensor:
        return s.pow(exponent=exponent)

    @staticmethod
    def derivative(s: Tensor, exponent: Tensor) -> Tensor:
        return exponent * s.pow(exponent=exponent - 1.)

    @staticmethod
    def forward(ctx: Any, M: Tensor, exponent: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_powm.value, exponent, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, exponent)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, exponent = ctx.saved_tensors
        dM = sym_modeig.backward(dX, s, smod, U, sym_powm.derivative, exponent)

        dXs = (U.transpose(-1, -2) @ ensure_sym(dX) @ U).diagonal(dim1=-1, dim2=-2)
        dexp = dXs * smod * s.log()

        return dM, dexp, None


class sym_sqrtm(F):
    """
    Computes the matrix square root for a batch of SPD matrices.
    """

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).sqrt()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        sder = 0.5 * s.rsqrt()
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_sqrtm.derivative), None


class sym_invsqrtm(F):
    """
    Computes the inverse matrix square root for a batch of SPD matrices.
    """

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).rsqrt()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        sder = -0.5 * s.pow(-1.5)
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invsqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invsqrtm.derivative), None


class sym_invsqrtm2(F):
    """
    Computes the square root and inverse square root matrices for a batch of SPD matrices.
    """

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        Xsq, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        smod2 = sym_invsqrtm.value(s)
        Xinvsq = U @ torch.diag_embed(smod2) @ U.transpose(-1, -2)
        ctx.save_for_backward(s, smod, smod2, U)
        return Xsq, Xinvsq

    @staticmethod
    def backward(ctx: Any, dXsq: Tensor, dXinvsq: Tensor):
        s, smod, smod2, U = ctx.saved_tensors
        dMsq = sym_modeig.backward(dXsq, s, smod, U, sym_sqrtm.derivative)
        dMinvsq = sym_modeig.backward(dXinvsq, s, smod2, U, sym_invsqrtm.derivative)

        return dMsq + dMinvsq, None


class sym_invm(F):
    """
    Computes the inverse matrices for a batch of SPD matrices.
    """

    @staticmethod
    def value(s: Tensor, param: Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).reciprocal()

    @staticmethod
    def derivative(s: Tensor, param: Tensor = None) -> Tensor:
        sder = -1. * s.pow(-2)
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric: bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invm.derivative), None
