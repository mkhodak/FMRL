import os
from operator import itemgetter
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from text_embedding.vectors import vocab2vecs


HOME = '/home/mak/'
DATA = HOME + 'FMRL/data/miniwikinet/'
VECDIR = HOME + 'glove.6B/'


def gaussian_mixture(ncls, m, dim):
    '''returns samples from a mixture of gaussians
    Args:
        ncls: number of Gaussians (classes)
        m: number of samples
        dim: data dimension
    Returns:
        numpy data array of shape [m, dim], numpy label array of shape [m,]
    '''

    X = normalize(np.random.normal(size=(m, dim)))
    mu = np.random.normal(size=(ncls, dim))
    Y = np.random.randint(4, size=m)
    for i, j in enumerate(Y):
        X[i] += mu[j]
    return X, Y


def word2vec(dim):
    '''returns dict of GloVe embeddings
    Args:
        dim: vector dimension (50, 100, 200, or 300)
    Returns:
        {word: vector} dict
    '''

    w2v = vocab2vecs(vectorfile=VECDIR + str(dim) + 'd.h5')
    w2v[0] = np.zeros(dim, dtype=np.float32)
    return w2v


def text2cbow(fname, w2v):
    '''returns CBOW text representations from file with "label\ttoken token ... token\n" on each line
    Args:
        fname: file name
        w2v: {word: vector} dict
    Returns:
        numpy data array of shape [number of lines, vector dimension], numpy label array of shape [number of lines,]
    '''

    z = w2v[0]
    with open(fname, 'r') as f:
        labels, texts = zip(*(line.strip().split('\t') for line in f))
        X = np.array([sum((w2v.get(w.lower(), z) for w in text.split()), np.copy(z)) for text in texts])
        nz = norm(X, axis=1) > 0.0
        X[nz] = normalize(X[nz])
        return X, np.array([int(label) for label in labels])


def textfiles(corpus='bal', partition='train', m=32):
    '''returns text file names
    Args:
        corpus: which subcorpus to use ('bal' or 'raw')
        partition: which partition to use ('train', 'dev', or 'test') ; ignored if corpus = 'raw'
        m: number of data points per class (1, 2, 4, ... , or 32) ; ignored if corpus = 'raw'
    Returns:
        list of filenames
    '''
    
    datadir = DATA+corpus+'/'
    if corpus == 'bal':
        datadir += partition+'/'+str(m)+'/'
    return [datadir+fname for fname, _ in sorted(((fname, int(fname[:-4])) for fname in os.listdir(datadir)), key=itemgetter(1))]
