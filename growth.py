import random
import h5py
import numpy as np
import torch
from numpy.linalg import norm
from torch import nn
from FMRL.data import gaussian_mixture
from FMRL.data import word2vec
from FMRL.data import textfiles
from FMRL.data import text2cbow
from FMRL.utils import ConstrainedLogit as CLogit
from FMRL.utils import MultiClassLinear
from FMRL.utils import OVALoss as OVAL
from FMRL.utils import project
from FMRL.utils import frank_wolfe


def growth(X, Y, radius=1.0, inc=0.02, verbose=False, tol=1E-3, **kwargs):
    '''computes growth of logit loss function with distance from optimum
    Args:
        X: numpy data array of size [number of samples, dimension]
        Y: numpy label array of size [number of samples,]
        radius: radius containing possible parameters
        inc: distance increment
        verbose: display output
        tol: unmodified frank_wolfe tolerance
        kwargs: passed to frank_wolfe
    Returns:
        numpy array of distances from optimum, numpy array of minimal function difference with optimal
    '''

    logit = CLogit(radius=radius, n_jobs=-1)
    logit.fit(X, Y)
    argmin = logit.coef_.flatten()
    ncls = len(np.unique(Y))

    weight = torch.Tensor(logit.coef_).float()
    model = MultiClassLinear(X.shape[1], ncls)
    Xtensor, Ytensor = torch.Tensor(X).float(), torch.Tensor(Y).long()
    criterion = OVAL(ncls, reduction='sum')
    def closure(grad=True):
        loss = criterion(model(Xtensor), Ytensor)
        if grad:
            model.zero_grad()
            loss.backward()
        return loss

    params = next(model.parameters())
    params.data = weight.clone()
    minval = float(criterion(model(Xtensor), Ytensor))
    deltas = np.arange(0.0, radius+inc, inc)
    epsilons = np.zeros(deltas.shape)

    for i, delta in enumerate(deltas):
        if i:
            b = (1.0 - delta**2 + norm(argmin)**2) / 2.0
            project(params.data, radius=radius, a=weight, b=b)
            frank_wolfe(closure, params, radius=radius, a=weight, b=b, tol=minval*tol, **kwargs)
            epsilons[i] = float(criterion(model(Xtensor), Ytensor)-minval)
            if verbose:
                print('\r', round(delta, 2), '\t', round(epsilons[i], 3), end='')
    if verbose:
        growth = min(epsilons[1:]/deltas[1:])
        print('\rMin Growth:     \t', round(growth, 6), '(Linear)\t', round(growth**2, 6), '(Quadratic)')
    return deltas, epsilons


def main():

    ncls, dim, verbose, n = 4, 50, True, 40

    kmin, kmax = 4, 9
    mixture_growth, mixture_samples = [], np.array([2**k for k in range(kmin, kmax+1)])
    np.random.seed(0)
    for _ in range(n):
        mixture_growth.append([])
        X, Y = gaussian_mixture(ncls, 2**kmax, dim)
        for m in mixture_samples:
            deltas, epsilons = growth(X[:m], Y[:m], verbose=verbose)
            mixture_growth[-1].append(epsilons)
    f = h5py.File('FMRL/mixture_growth.h5', 'w')
    f.create_dataset('m', data=mixture_samples)
    f.create_dataset('delta', data=deltas)
    f.create_dataset('epsilon', data=np.array(mixture_growth))
    f.close()

    w2v = word2vec(dim)
    kmin, kmax = 0, 5
    cbow_growth, cbow_samples = [], np.array([2**k for k in range(kmin, kmax+1)])
    fnames = textfiles(m=2**kmax)
    random.seed(0)
    for _ in range(n):
        cbow_growth.append([])
        fname = fnames[random.randint(0, len(fnames))].split('/')
        for m in cbow_samples:
            fname[-2] = str(m)
            X, Y = text2cbow('/'.join(fname), w2v)
            deltas, epsilons = growth(X, Y, verbose=verbose)
            cbow_growth[-1].append(epsilons)
    f = h5py.File('FMRL/cbow_growth.h5', 'w')
    f.create_dataset('m', data=cbow_samples)
    f.create_dataset('delta', data=deltas)
    f.create_dataset('epsilon', data=np.array(cbow_growth))
    f.close()


if __name__ == '__main__':

   main() 
