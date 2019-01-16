import pdb
import numpy as np
import torch
from numpy.linalg import norm
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.preprocessing import normalize
from torch import nn


class ConstrainedLogit(Logit):

    def __init__(self, radius=1.0, **kwargs):

        self.radius = 1.0
        self.kwargs = kwargs

    def logit(self, logC, X, Y):

        super().__init__(C=10**logC, fit_intercept=False, **self.kwargs)
        super().fit(X, Y)
        return norm(self.coef_) - self.radius

    def fit(self, X, Y):

        brentq(self.logit, -4, 4, args=(X, Y))


class MultinomialLogit(nn.Module):

    def __init__(self, dim, ncls, init=None):
        
        super().__init__()
        self.linear = nn.Linear(dim, ncls, bias=False)
        if not init is None:
            self.linear.weight.data = init

    def forward(self, input):

        return self.linear(input), next(self.parameters())


def cuda_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def growth(X, Y, radius=1.0, delta_inc=0.05, rho_inc=0.05, tol=1E-4, **kwargs):

    logit = ConstrainedLogit(radius=radius, **kwargs)
    logit.fit(X, Y)

    deltas = np.arange(delta_inc, radius+delta_inc, delta_inc)
    funcvals = np.inf * np.ones(deltas.shape)
    optnormsq = norm(logit.coef_)**2

    device = cuda_device()
    weight = torch.Tensor(logit.coef_).float().to(device)
    model = MultinomialLogit(X.shape[1], len(np.unique(Y)), init=None).to(device)
    Xtensor, Ytensor = torch.Tensor(X).float().to(device), torch.Tensor(Y).long().to(device)
    
    pdb.set_trace()

    relu = lambda input: nn.functional.relu(input)
    eq = lambda params: (params*params).sum() - radius

    for i, delta in enumerate(deltas):
        for rho in np.arange(rho_inc, radius+rho_inc, rho_inc):
            bias = delta**2 - rho**2 - optnormsq
            ineq = lambda params: 2.0 * (weight*params).sum() + bias
            obj = lambda lambda1, lambda2, logits, params: lambda1*relu(ineq(params)) + lambda2*relu(eq(params))
            lambda1, lambda2 = 1.0, 1.0
            for _ in range(100):
                closure = lambda: obj(lambda1, lambda2, *model(Xtensor))
                #optimizer = torch.optim.LBFGS(model.parameters())
                optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
                loss = closure()
                loss.backward()
                optimizer.step()
                params = next(model.parameters())
                print('Ineq:', float(ineq(params)), 'Eq:', float(eq(params)))
                if relu(ineq(params)) > tol:
                    lambda1 *= 2.0
                elif relu(eq(params)) > tol:
                    lambda2 *= 2.0
            pdb.set_trace()


if __name__ == '__main__':

    X = normalize(np.random.normal(size=(128, 50)))
    mu = np.random.normal(size=(4, 50))
    Y = np.random.randint(4, size=128)
    for i, j in enumerate(Y):
        X[i] += mu[j]

    growth(X, Y, n_jobs=-1)
    pdb.set_trace()
