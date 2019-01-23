import pdb
import sys
from math import sqrt
import cvxpy as cp
import h5py
import numpy as np
import torch
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression as Logit
sys.path.append('/home/ubuntu')
from FMRL.data import word2vec
from FMRL.data import textfiles
from FMRL.data import text2cbow
from FMRL.utils import ConstrainedLogit as CLogit
from FMRL.utils import MultiClassLinear
from FMRL.utils import OVALoss as OVAL
from FMRL.utils import frank_wolfe


class BiasRegularizedLogit(Logit):

    def __init__(self, radius=1.0, eta=None, phi=None):

        self.radius = radius
        self.eta = eta
        self.phi = phi
        super().__init__(fit_intercept=False)
        self.coef_ = None

    def fit(self, X, Y, ncls=None):

        if ncls is None:
            ncls = len(np.unique(Y))
        theta = cp.Variable((ncls, X.shape[1]))
        if self.eta is None:
            target = 0.0
        else:
            target = 0.5/self.eta * cp.sum_squares(theta-self.phi)
        for i in range(ncls):
            M = np.dot(1.0-2.0*(Y==i), X)
            target += cp.sum(cp.logistic(M*theta[i]))
        objective = cp.Minimize(target)
        constraints = [cp.sum_squares(theta) <= self.radius**2]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.coef_ = theta.value


class Baseline(CLogit):

    def __init__(self, model, loss, radius=1.0, **kwargs):

        self.model = model
        self.loss = loss
        self.radius = radius
        self.D = radius
        self.params = next(self.model.parameters())
        self.phi = self.params.data.clone()
        super().__init__(radius=radius, **kwargs)

    def get_closure(self, X, Y, eta, phi):

        model = self.model
        params = self.params
        coef = 0.5 / eta
        def closure(grad=True):
            div = params - phi
            loss = self.loss(model(X), Y) + coef*(div*div).sum()
            if grad:
                model.zero_grad()
                loss.backward()
            return loss
        return closure

    def ftrl(self, X, Y):

        params = self.params
        params.data = self.phi.clone()
        m = X.shape[0]
        eta = self.D / sqrt(m)
        losses = np.empty(m)
        blogit = BiasRegularizedLogit(radius=self.radius, eta=eta, phi=self.phi.detach().numpy())
        Xarray, Yarray = X.detach().numpy(), Y.detach().numpy()
        for i in range(1, m+1):
            losses[i-1] = float(self.loss(self.model(X[i-1:i]), Y[i-1:i]))
            try:
                blogit.fit(Xarray[:i], Yarray[:i], ncls=4)
                params.data = torch.Tensor(blogit.coef_)
            except cp.error.SolverError:
                closure = self.get_closure(X[:i], Y[:i], eta, self.phi)
                frank_wolfe(closure, params, radius=self.radius)
        return losses

    def ogd(self, X, Y):

        params = self.params
        params.data = self.phi.clone()
        m = X.shape[0]
        eta = self.D / sqrt(m)
        losses = np.empty(m)
        for i in range(1, m+1):
            loss = self.loss(self.model(X[i-1:i]), Y[i-1:i])
            self.model.zero_grad()
            loss.backward()
            params.data -= eta*params.grad
            magnitude = torch.norm(params.data)
            if magnitude > self.radius:
                params.data *= self.radius / magnitude
            losses[i-1] = float(loss)
        return losses

    def meta(self, X, Y, method='ogd'):

        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        losses = getattr(self, method)(Xtensor, Ytensor)
        self.fit(X, Y)
        self.params.data = torch.Tensor(self.coef_)
        return losses.sum() - float(self.loss(self.model(Xtensor), Ytensor))


class Strawman(Baseline):

    def __init__(self, *args, D=0.1, gamma=1.1, **kwargs):

        super().__init__(*args, **kwargs)
        self.t = 0
        self.D = D
        self.gamma = gamma

    def meta(self, X, Y, method='ogd'):

        self.t += 1
        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        losses = getattr(self, method)(Xtensor, Ytensor)
        self.fit(X, Y)
        opt = torch.Tensor(self.coef_)
        self.params.data = opt
        regret = losses.sum() - float(self.loss(self.model(Xtensor), Ytensor))
        opt = torch.Tensor(self.coef_)
        if self.t > 1 and torch.norm(opt-self.phi) > self.D:
            self.D *= self.gamma
        self.phi = opt.clone()
        return regret


class FML(Strawman):

    def meta(self, X, Y, method='ogd'):

        self.t += 1
        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        losses = getattr(self, method)(Xtensor, Ytensor)
        self.fit(X, Y)
        opt = torch.Tensor(self.coef_)
        self.params.data = opt
        regret = losses.sum() - float(self.loss(self.model(Xtensor), Ytensor))
        opt = torch.Tensor(self.coef_)
        if self.t > 1 and torch.norm(opt-self.phi) > self.D:
            self.D *= self.gamma
        self.phi = (1-1/self.t)*self.phi + 1/self.t*opt
        return regret


class FLI(Strawman):

    def meta(self, X, Y, method='ogd'):

        self.t += 1
        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        losses = getattr(self, method)(Xtensor, Ytensor)
        last = self.params.data.clone()
        self.fit(X, Y)
        opt = torch.Tensor(self.coef_)
        self.params.data = opt
        regret = losses.sum() - float(self.loss(self.model(Xtensor), Ytensor))
        opt = torch.Tensor(self.coef_)
        if self.t > 1 and torch.norm(opt-self.phi) > self.D:
            self.D *= self.gamma
        self.phi = (1-1/self.t)*self.phi + 1/self.t*last
        return regret


def main():

    ncls, dim, verbose = 4, 50, True
    w2v = word2vec(dim)
    model = MultiClassLinear(dim, ncls)
    params = next(model.parameters())
    loss = OVAL(ncls, reduction='sum')
    meta, task = sys.argv[1:]
    algos = {'baseline': Baseline, 'omniscient': Baseline, 'strawman': Strawman, 'fml': FML, 'fli': FLI}

    f = h5py.File('FMRL/'+meta+'-'+task+'-online.h5', 'w')
    for k in range(0, 6):
        m = 2**k
        print('\rComputing Regret of', m, 'Shot Classification')
        fnames = textfiles(m=m)
        params.data *= 0.0
        algo = algos[sys.argv[1]](model, loss)
        if meta == 'omniscient':
            g = h5py.File('FMRL/cbow_similarity.h5')
            opt = np.array(g[str(m)])
            g.close()
            mean = opt.mean(0)
            algo.phi = torch.Tensor(mean.reshape(ncls, dim))
            algo.D = max(norm(theta-mean) for theta in opt)
        regret = []
        guesses = []
        for i, fname in enumerate(fnames):
            guesses.append(algo.D)
            X, Y = text2cbow(fname, w2v)
            regret.append(algo.meta(X, Y, method=task))
            if verbose:
                print('\rProcessed', i+1, 'Tasks', end='')
                print('; TAR:', round(np.mean(regret), 5), end='')
        print()
        f.create_dataset(str(m), data=np.array([regret, guesses]))
    f.close()

if __name__ == '__main__':

    main()
