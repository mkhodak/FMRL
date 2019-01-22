import pdb
import sys
from math import sqrt
import cvxpy as cp
import h5py
import numpy as np
import torch
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression as Logit
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
    '''baseline single-task OCO'''

    def __init__(self, model, loss, radius=1.0, **kwargs):

        self.model = model
        self.loss = loss
        self.radius = radius
        self.D = 2.0 * radius
        self.t = 0
        super().__init__(radius=radius, **kwargs)

    def get_closure(self, X, Y, eta, phi):

        model = self.model
        params = next(model.parameters())
        coef = 0.5 / eta
        def closure(grad=True):
            div = params - phi
            loss = self.loss(model(X), Y) + coef*(div*div).sum()
            if grad:
                model.zero_grad()
                loss.backward()
            return loss
        return closure

    def ftrl(self, X, Y, store=False, batch=False):

        params = next(self.model.parameters())
        m = X.shape[0]
        eta = self.D / sqrt(m)
        phi = params.data.clone()
        losses = np.empty(m)
        blogit = BiasRegularizedLogit(radius=self.radius, eta=eta, phi=phi.detach().numpy())
        Xarray, Yarray = X.detach().numpy(), Y.detach().numpy()
        for i in range(1, m+1):
            if batch and i < m and not store:
                continue
            if store:
                if i == 1:
                    avg = params.data.clone()
                else:
                    avg += params.data
            losses[i-1] = float(self.loss(self.model(X[i-1:i]), Y[i-1:i]))
            try:
                blogit.fit(Xarray[:i], Yarray[:i], ncls=4)
                params.data = torch.Tensor(blogit.coef_)
            except cp.error.SolverError:
                closure = self.get_closure(X[:i], Y[:i], eta, phi)
                frank_wolfe(closure, params, radius=self.radius)
            if torch.norm(params.data) > self.radius and not np.isclose(torch.norm(params.data), self.radius):
                pdb.set_trace()
        if store:
            params.data = avg / m
        return losses

    def ogd(self, X, Y, store=False, **kwargs):

        params = next(self.model.parameters())
        m = X.shape[0]
        eta = self.D / sqrt(m)
        losses = np.empty(m)
        for i in range(1, m+1):
            if store:
                if i == 1:
                    avg = params.data.clone()
                else:
                    avg += params.data
            loss = self.loss(self.model(X[i-1:i]), Y[i-1:i])
            self.model.zero_grad()
            loss.backward()
            params.data -= eta*params.grad
            normp = torch.norm(params.data)
            if normp > self.radius:
                params.data *= self.radius / normp
            losses[i-1] = float(loss)
        if store:
            params.data = avg / m
        return losses

    def meta(self, X, Y, method='ftrl'):

        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        self.t += 1
        params = next(self.model.parameters())
        if self.t > 1:
            params.data = self.phi.clone()
        else:
            self.phi = params.data.clone()
        losses = getattr(self, method)(Xtensor, Ytensor)
        self.fit(X, Y)
        params.data = torch.Tensor(self.coef_)
        return losses.sum() - float(self.loss(self.model(Xtensor), Ytensor))


class Strawman(Baseline):

    def __init__(self, model, loss, D=0.1, gamma=1.1, **kwargs):

        super().__init__(model, loss, **kwargs)
        self.D = D
        self.gamma = gamma

    def meta(self, X, Y, method='ftrl'):

        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        self.t += 1
        params = next(self.model.parameters())
        self.fit(X, Y)
        opt = torch.Tensor(self.coef_)
        if self.t > 1 and torch.norm(params-opt) > self.D:
            self.D *= self.gamma
        losses = getattr(self, method)(Xtensor, Ytensor)
        params.data = opt
        return losses.sum() - float(self.loss(self.model(Xtensor), Ytensor))


class FML(Strawman):

    def __init__(self, model, loss, aogd=False, **kwargs):

        super().__init__(model, loss, **kwargs)
        self.aogd = aogd
        self.sqrtm1t = 0

    def ftl_update(self, prev, m):

        params = next(self.model.parameters())
        sqrtm = sqrt(m)
        if self.t > 1:
            params.data *= sqrtm
            params.data += self.sqrtm1t * prev
            self.sqrtm1t += sqrtm
            params.data /= self.sqrtm1t
        else:
            self.sqrtm1t += sqrtm

    def aogd_update(self, prev, m):

        params = next(self.model.parameters())
        self.sqrtm1t += sqrt(m)
        if self.t > 1:
            params.data = prev - 1.0/self.sqrtm1t * (prev-params.data)

    def meta(self, X, Y, method='ftrl', batch=False):

        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        params = next(self.model.parameters())
        prev = params.data.clone()
        self.fit(X, Y)
        opt = torch.Tensor(self.coef_)
        if self.t and torch.norm(prev - opt) > self.D:
            self.D *= self.gamma
        if not batch:
            losses = getattr(self, method)(Xtensor, Ytensor, batch=batch)
        params.data = opt
        comp = float(self.loss(self.model(Xtensor), Ytensor))
        self.aogd_update(prev, X.shape[0]) if self.aogd else self.ftl_update(prev, X.shape[0])
        self.t += 1
        if batch:
            return None
        return losses.sum() - comp


class FLI(FML):

    def meta(self, X, Y, method='ftrl', avg=False, **kwargs):

        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        params = next(self.model.parameters())
        prev = params.data.clone()
        self.fit(X, Y)
        losses = getattr(self, method)(Xtensor, Ytensor, store=avg, **kwargs)
        update = params.data.clone()
        if self.t and torch.norm(prev - update) > self.D:
            self.D *= self.gamma
        opt = torch.Tensor(self.coef_)
        params.data = opt
        comp = float(self.loss(self.model(Xtensor), Ytensor))
        params.data = update
        self.aogd_update(prev, X.shape[0]) if self.aogd else self.ftl_update(prev, X.shape[0])
        self.t += 1
        return losses.sum() - comp


def main():

    ncls, dim, verbose = 4, 50, True
    w2v = word2vec(dim)
    model = MultiClassLinear(dim, ncls)
    params = next(model.parameters())
    loss = OVAL(ncls, reduction='sum')

    meta, task = sys.argv[1:]
    kwargs = {'method': task}
    if meta in {'baseline', 'omniscient'}:
        algo = Baseline(model, loss)
    elif meta == 'strawman':
        algo = Strawman(model, loss)
    elif meta == 'fml':
        algo = FML(model, loss)
    elif meta == 'fli':
        algo = FLI(model, loss)
        kwargs['avg'] = False
    else:
        raise(NotImplementedError)

    f = h5py.File('FMRL/'+task+'-'+meta+'-online.h5')
    for k in range(0, 6):
        m = 2**k
        print('\rComputing Regret of', m, 'Shot Classification')
        fnames = textfiles(m=m)
        if meta == 'omniscient':
            g = h5py.File('FMRL/cbow_similarity.h5')
            opt = np.array(g[str(m)])
            g.close()
            mean = opt.mean(0)
            params.data = torch.Tensor(mean.reshape(ncls, dim))
            algo.D = max(norm(theta-mean) for theta in opt)
        else:
            params.data *= 0.0
        regret = []
        for i, fname in enumerate(textfiles(m=m)):
            X, Y = text2cbow(fname, w2v)
            regret.append(algo.meta(X, Y, **kwargs))
            if verbose and not (i+1) % 10:
                print('\rProcessed', i+1, 'Tasks', end='')
                print(' ; TAR:', round(np.mean(regret), 5), end='')
        print('\rProcessed', i+1, 'Tasks ; TAR:', round(np.mean(regret), 5))
        f.create_dataset(str(m), data=np.array(regret))
    f.close()


if __name__ == '__main__':

    main()
