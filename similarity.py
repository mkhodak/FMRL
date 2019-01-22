import pdb
import h5py
import numpy as np
from FMRL.data import word2vec
from FMRL.data import textfiles
from FMRL.data import text2cbow
from FMRL.utils import ConstrainedLogit as CLogit


def main():

    ncls, dim, verbose = 4, 50, True
    w2v = word2vec(dim)
    kmin, kmax = 0, 5
    f = h5py.File('FMRL/cbow_similarity.h5')
    for m, corpus in zip([2**k for k in range(kmin, kmax+1)]+['all'], ['bal']*(kmax+1-kmin)+['raw']):
        fnames = textfiles(corpus=corpus, m=m)
        opt = []
        for i, fname in enumerate(fnames):
            X, Y = text2cbow(fname, w2v)
            logit = CLogit(n_jobs=-1)
            logit.fit(X, Y)
            opt.append(logit.coef_.flatten())
            if verbose and not (i+1) % 10:
                print('\rProcessed', i+1, 'Tasks', end='')
        f.create_dataset(str(m), data=np.array(opt))
        if verbose:
            print('\rProcessed '+str(m)+'-Shot Tasks')
    f.close()


if __name__ == '__main__':

    main()
