# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import numpy as np

def get_array_module(x):
    return np

def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if word.strip() == '':
            word2 = str(word.encode("utf-8"))
            print(f'Warning: only space chars in word ({word2})', file=sys.stderr)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


def write(words, matrix, file):
    m = matrix
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)

            
def length_normalize(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]

def mean_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg

def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
#################### End of original code ##################################
#################### Start of new code ##################################
        else:
            all = globals()
            from inspect import isfunction
            if action in all and isfunction(all[action]):
                all[action](matrix)
            else:
                raise Exception('Unknown action: ' + action)

def sqrt(matrix):
    xp = get_array_module(matrix)
    matrix[:] = xp.sqrt(matrix)

def median_center(matrix):
    xp = get_array_module(matrix)
    # m = xp.median(matrix, axis=0)
    m = np.percentile(matrix, q=50, axis=0)
    matrix -= m

def pmi(X):
    eps = 1e-8
    rs = X.sum(axis=0, keepdims=True)
    cs = X.sum(axis=1, keepdims=True)
    X /= rs + eps
    X /= cs + eps

def levy2014k(X, k=1):
    eps = 1e-8
    sum1 = np.sum(np.abs(X), axis=1, keepdims=True) + eps
    sum0 = np.sum(np.abs(X), axis=0, keepdims=True) + eps
    N = np.sum(X)
    X[:] = np.maximum(0, np.log(X) + np.log(N) - np.log(sum1) - np.log(sum0) - np.log(k))

def levy2014_k5(X):
    levy2014k(X, k=5)

def levy2014(X):
    levy2014k(X, k=1)


def log(X):
    X[:] = np.maximum(0, np.log(X))

def log1p(X):
    X[:] = np.log(1 + X)

def glove(X):
    # (8) of the glove paper: https://aclanthology.org/D14-1162.pdf
    Y = np.log(1+X)
    for _ in range(5):
        bi = np.mean(Y, axis=1, keepdims=True)
        Y -= bi 
        bj = np.mean(Y, axis=0, keepdims=True)
        Y -= bj
        print('bi ', np.mean(np.abs(bi)))
    if np.mean(np.abs(bi)) > 1e-6:
        print('bi failed', np.mean(np.abs(bi)))
    if np.mean(np.abs(bj)) > 1e-6:
        print('bj failed', np.mean(np.abs(bj)))
    X[:] = Y

def unitL1(X):
    norm1 = np.sum(np.abs(X), axis=1, keepdims=True)
    norm1[norm1 == 0] = 1 
    X /= norm1 

def fung1997(X):
    from scipy.special import xlogy
    sum1 = np.sum(np.abs(X), axis=1, keepdims=True)
    sum0 = np.sum(np.abs(X), axis=0, keepdims=True)
    N = np.sum(X)
    X[:] = xlogy(X / N, X * N / (sum1 * sum0))


def length_normalize_axis0(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms

def mean_center_axis1(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=1)
    matrix -= avg[:, xp.newaxis]


# import faiss
# def faiss_knn(Q, X, k, dist='IP'):
#     d = X.shape[1]
#     if dist == 'IP':
#         index = faiss.IndexFlatIP(d)
#     elif dist == 'L2':
#         index = faiss.IndexFlatL2(d)
#     index.add(X)
#     dists, inds = index.search(Q, k)
#     return dists, inds

# def faiss_csls(Q, X, k, dist='IP', csls=10):
#     # this k is neighborhood
#     sim_bwd, _ = faiss_knn(X, Q, k=csls)
#     knn_sim_bwd = sim_bwd.mean(axis=1)
#     topvals, topinds = faiss_knn(Q, X, k=2*csls)
#     for i in range(topvals.shape[0]):
#         topvals[i] = 2 * topvals[i] - knn_sim_bwd[topinds[i]]
#     ind = (-topvals).argsort(axis=1)
#     topvals = np.take_along_axis(topvals, ind, axis=1)
#     topinds = np.take_along_axis(topinds, ind, axis=1)
#     return topvals, topinds

# def noise(X):
#     xp = get_array_module(X)
#     noise = np.random.randn(1, X.shape[1])
#     noise /= xp.sqrt(xp.sum(noise**2))
#     # size = np.random.randint(1, 3)
#     size = 1 
#     randinds = np.random.randint(X.shape[1], size=size)
#     X -= np.mean(X[randinds, :], axis=0)
#     normalize(X, ['unit', 'center', 'unit'])


# def joint_noise(X, Y):
#     xp = get_array_module(X)
#     noise = np.random.randn(1, X.shape[1])
#     noise /= xp.sqrt(xp.sum(noise**2))
#     randinds = np.random.randint(X.shape[1], size=1)
    
#     randcenter = np.mean(X[randinds, :], axis=0)
#     X -= randcenter
#     Y -= randcenter
#     normalize(X, ['unit', 'center', 'unit'])
#     normalize(Y, ['unit', 'center', 'unit'])
