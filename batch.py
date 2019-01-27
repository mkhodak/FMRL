import pdb
import random
import sys
from copy import deepcopy
from itertools import combinations
import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression as Logit
from FMRL.data import word2vec
from FMRL.data import textfiles
from FMRL.data import text2cbow
from FMRL.online import FML
from FMRL.online import FLI
from FMRL.utils import MultiClassLinear
from FMRL.utils import OVALoss as OVAL


class FMRL:

    def __init__(self, model, loss, variant='fml', **kwargs):

        if variant == 'fml':
            self.learner = FML(model, loss, **kwargs)
        elif variant == 'fli':
            self.learner = FLI(model, loss, **kwargs)
        else:
            raise(NotImplementedError)
        self.meta = self.learner.meta

    def train(self, X, Y, method='ftrl', avg=True, **kwargs):

        Xtensor, Ytensor = torch.Tensor(X), torch.Tensor(Y)
        params = next(self.learner.model.parameters())
        fixed = params.data.clone()
        getattr(self.learner, method)(Xtensor, Ytensor, store=avg, **kwargs)
        out = params.detach().numpy()
        params.data = fixed
        return out


class MAML:

    def __init__(self, model, loss, radius=1.0, eta_meta=0.01, eta_task=0.01):
        self.model = model
        self.loss = loss
        self.radius = radius
        self.eta_meta = eta_meta
        self.eta_task = eta_task

    def project(self):

        params = next(self.model.parameters())
        normp = torch.norm(params.data)
        if normp > self.radius:
            params.data *= self.radius / normp

    def closure(self, X, Y):

        loss = self.loss(self.model(X), Y)
        self.model.zero_grad()
        loss.backward()

    def meta(self, X, Y):

        model, loss = self.model, self.loss
        train = int(X.shape[0]/2)
        self.closure(torch.Tensor(X[:train]), torch.Tensor(Y[:train]))
        params = next(model.parameters())
        prev = params.data.clone()
        params.data -= self.eta_task * params.grad
        self.project()
        self.closure(torch.Tensor(X[train:]), torch.Tensor(Y[train:]))
        params.data = prev - self.eta_meta * params.grad
        self.project()

    def train(self, X, Y):

        model, loss = self.model, self.loss
        self.closure(torch.Tensor(X), torch.Tensor(Y))
        params = next(model.parameters())
        fixed = params.data.clone()
        params.data -= self.eta_task * params.grad
        self.project()
        out = params.detach().numpy()
        params.data = fixed
        return out


def main():

    ncls, dim, verbose = 4, 50, True
    w2v = word2vec(dim)
    model = MultiClassLinear(dim, ncls)
    params = next(model.parameters())
    params.data *= 0.0
    loss = OVAL(ncls, reduction='sum')
    random.seed(0)
    options = [4**-i for i in range(6)]
    #options = [1E-3, 1E-2, 1E-1, 1E0]
    
    
    meta = sys.argv[1]
    if meta == 'maml':
        sweep = [{'eta_meta': eta_meta, 'eta_task': eta_task} for eta_meta, eta_task in combinations(options, 2)]
        cls = MAML
        kwargs = {}
        trargs = kwargs
    else:
        sweep = [{'variant': meta, 'D': D, 'gamma': gammam1+1.0} for D, gammam1 in combinations(options, 2)]
        cls = FMRL
        task = sys.argv[2]
        kwargs = {'method': task}
        if task == 'ftrl':
            kwargs['batch'] = True
        try:
            trargs = {'avg': not sys.argv[3] == 'last'}
            trargs.update(kwargs)
        except IndexError:
            trargs = kwargs

    f = h5py.File('FMRL/'+'-'.join(sys.argv[1:])+'-batch.h5')
    for k in range(0, 6):
        m = 2**k
        print('\rTraining', m, 'Shot Classification')
        fnames = textfiles(m=m)
        random.shuffle(fnames)
        best, bestacc = None, 0.0
        for etas in sweep:
            algo = cls(model, loss, **etas)
            for i, fname in enumerate(fnames):
                X, Y = text2cbow(fname, w2v)
                algo.meta(X, Y, **kwargs)
                if verbose and not (i+1) % 10:
                    print('\rTrained on', i+1, 'Tasks', end='')
            clf = Logit(fit_intercept=False, n_jobs=-1)
            clf.fit(X, Y)
            acc = []
            for i, (train, test) in enumerate(zip(textfiles(partition='dev', m=m), textfiles(partition='dev', m='test'))):
                X, Y = text2cbow(train, w2v)
                clf.coef_ = algo.train(X, Y, **trargs)
                X, Y = text2cbow(test, w2v)
                acc.append(clf.score(X, Y))
                if verbose and not (i+1) % 10:
                    print('\rValidated on', i+1, 'Tasks', end='')
                    print(' ; Acc:', np.round(np.mean(acc), 5), end='')
            if np.mean(acc) > bestacc:
                best = deepcopy(algo), params.data.clone()
                bestacc = np.mean(acc)
        algo = best[0]
        params.data = best[1]
        acc = []
        for i, (train, test) in enumerate(zip(textfiles(partition='test', m=m), textfiles(partition='test', m='test'))):
            X, Y = text2cbow(train, w2v)
            clf.coef_ = algo.train(X, Y, **trargs)
            X, Y = text2cbow(test, w2v)
            acc.append(clf.score(X, Y))
            if verbose and not (i+1) % 10:
                print('\rEvaluated on', i+1, 'Tasks', end='')
                print(' ; Acc:', np.round(np.mean(acc), 5), end='')
        print('\rEvaluated on', i+1, 'Tasks ; Acc:', np.round(np.mean(acc), 5))
        f.create_dataset(str(m), data=np.array(acc))
    f.close()


if __name__ == '__main__':

    main()
