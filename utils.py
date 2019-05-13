import cvxpy as cp
import numpy as np
import torch
from numpy.linalg import norm
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression as Logit
from torch import nn


def project(x, radius=1.0, a=None, b=float('inf')):
    '''projects point in place according to l2-norm and an inequality constraint
    Args:
        x: torch.FloatTensor
        radius: radius of l2-norm constraint
        a: torch.FloatTensor of same shape as 'x' ; constraint <a,x> <= b
        b: float ; constraint <a,x> <= b
    '''

    if not a is None:
        aTx = (a*x).sum()
        if aTx > b:
            x -= (aTx-b) / (a*a).sum() * a
    if radius < float('inf'):
        normx = torch.norm(x)
        if normx > radius:
            x *= radius / normx


def frank_wolfe(closure, params, radius=1.0, a=None, b=float('inf'), tol=1E-3, max_iter=1000, ls=10):
    '''constrained optimization under l2-norm and an inequality constraint using Frank-Wolfe algorithm
    Args:
        closure: PyTorch model evaluation and loss computation
        params: PyTorch model parameters
        radius: radius of l2-norm constraint
        a: torch.FloatTensor of same shape as 'params' ; constraint <a,x> <= b
        b: float ; constraint <a,x> <= b
        max_iter: maximum number of iterations
        tol: absolute tolerance
        ls: number of steps of linesearch
    '''

    ncls, dim = params.shape
    if not a is None:
        A = a.numpy().flatten()
        s = cp.Variable(ncls*dim)
    for i in range(max_iter):
        loss = closure()
        if a is None:
            s = -(radius/torch.norm(params.grad)) * params.grad
            u = s - params
        else:
            grad = params.grad.numpy().flatten()
            objective = cp.Minimize(cp.sum(s*grad))
            constraints = [cp.norm(s) <= radius]
            if not a is None:
                constraints.append(A*s <= b)
            prob = cp.Problem(objective, constraints)
            prob.solve()
            u = torch.Tensor(s.value.reshape(ncls, dim)) - params
        gap = -float((params.grad*u).sum())
        if gap < tol:
            break
        best = float(loss)
        eta = 2.0/(2.0+i) / ls
        for _ in range(ls):
            params.data = params + eta*u
            current = float(closure(grad=False))
            if current < best:
                best = current
            else:
                params.data = params.data - 0.5*eta*u
                break
    else:
        print('Did Not Converge', gap)


class ConstrainedLogit(Logit):
    '''l2-constrained logit using root-finding'''

    def __init__(self, radius=1.0, interval=[-4, 4], **kwargs):
        '''
        Args:
            radius: l2-norm constraint
            interval: root-finding interval
            kwargs: passed to Logit
        '''

        self.radius = radius
        self.interval = interval
        self.kwargs = kwargs

    def logit(self, logC, X, Y):
        '''fits unconstrained l2-regularized logit
        Args:
            logC: logarithm of regularization parameter
            X: samples
            Y: labels
        Returns:
            difference between parameter norm and l2-constraint
        '''

        super().__init__(C=10**logC, fit_intercept=False, **self.kwargs)
        super().fit(X, Y)
        return norm(self.coef_) - self.radius

    def fit(self, X, Y):
        '''fits constrained logit
        Args:
            X: samples
            Y: labels
        '''
        
        brentq(self.logit, *self.interval, args=(X, Y))


class MultiClassLinear(nn.Linear):
    '''multi-class classifier PyTorch module'''

    def __init__(self, dim, ncls):
        '''
        Args:
            dim: data dimension
            ncls: number of classes
        '''

        super().__init__(dim, ncls, bias=False)


class OVALoss(nn.BCEWithLogitsLoss):
    '''multi-class one-v-all loss'''

    def __init__(self, ncls, **kwargs):
        '''
        Args:
            ncls: number of classes
        '''

        self.ncls = ncls
        super().__init__(**kwargs)

    def forward(self, input, target):
        '''
        Args:
            input: torch.FloatTensor of shape [number of samples, self.ncls]
            target: torch.FloatTensor of shape [number of samples] containing class labels
        Returns:
            PyTorch loss
        '''

        logloss = super().forward
        loss = logloss(input[:,0], torch.eq(target, 0).float())
        for cls in range(1, self.ncls):
            loss += logloss(input[:,cls], torch.eq(target, cls).float())
        return loss
