# Copyright (c) Meta Platforms, Inc. and affiliates.
from collections import Counter
import numpy as np
import embeddings

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
MAX_SVD_DIM = 5000 # maximum SVD to avoid long compute time

### initialization methods ###
def vecmap_unsup(x, z, norm_proc=['unit', 'center', 'unit']):
    print('maxdim', MAX_SVD_DIM)
    sim_size = min(MAX_SVD_DIM, min(x.shape[0], z.shape[0]))
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    xsim = (u*s).dot(u.T)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    zsim = (u*s).dot(u.T)
    del u, s, vt
    xsim.sort(axis=1)
    zsim.sort(axis=1)
    norm_proc = ['unit', 'center', 'unit']
    embeddings.normalize(xsim, norm_proc)
    embeddings.normalize(zsim, norm_proc)
    sim = xsim.dot(zsim.T)
    return sim

def match_sim(xsim, zsim, sort=True, metric='cosine', norm_proc=['unit', 'center', 'unit']):
    sim_size = min(xsim.shape[1], zsim.shape[1])
    xsim = np.array(xsim[:, :sim_size])
    zsim = np.array(zsim[:, :sim_size])
    if sort:
        xsim.sort(axis=1)
        zsim.sort(axis=1)
    embeddings.normalize(xsim, norm_proc)
    embeddings.normalize(zsim, norm_proc)
    sim = xsim @ zsim.T 
    return sim

### main search loops ###
def vecmap(x: np.ndarray, z: np.ndarray, args, sim_init=None, evalf=None):
    print('running vecmap', x.shape)
    keep_prob = args.stochastic_initial
    best_objective = float('-inf')
    last_improvement = 0
    end = False
    inds1, inds2 = 0, 0
    for it in range(args.maxiter):
        if it - last_improvement > args.stochastic_interval:
            # maxswaps = max(1, maxswaps - 1)
            if keep_prob == 1:
                end = True
            keep_prob = min(1.0, args.stochastic_multiplier * keep_prob)
            last_improvement = it

        if it == 0:
            if sim_init is not None:
                sim = sim_init
            else:
                sim = vecmap_unsup(x, z, norm_proc=['unit', 'center', 'unit'])
        else:
            # rotation
            if args.method == 'orthogonal':
                u, s, vt = np.linalg.svd(x[inds1].T @ z[inds2])
                w = u @ vt
            elif args.method == 'lstsq':
                w, r, r, s = np.linalg.lstsq(x[inds1], z[inds2], rcond=1e-5)
            sim = x @ w @ z.T

        # 
        if args.csls:
            sim = most_diff_match(sim, 10)
        inds1, inds2, evalsim = match(sim, args.match)

        if evalf is not None:
            evalf(evalsim)

        objf = np.mean(sim.max(axis=1))
        objb = np.mean(sim.max(axis=0))

        objective = (objf + objb) / 2
        print(f'{it} {keep_prob} \t{objf:.4f}\t{objective:.4f}\t{best_objective:.4f}')

        if objective >= best_objective + args.threshold:
            last_improvement = it
            if it != 0:
                best_objective = objective

        if end:
            break
    return inds1, inds2, sim

def coocmapt(Cp1: np.ndarray, Cp2: np.ndarray, args, normproc=['unit'], sim_init=None, evalf=None):
    """
    basic coocmap using just numpy but only works for cosine distance
    """
    best_objective = float('-inf')
    last_improvement = 0
    end = False
    inds1, inds2 = 0, 0
    simd = 0
    for it in range(args.maxiter):
        if it - last_improvement > args.stochastic_interval:
            end = True

        if it == 0:
            if sim_init is not None:
                sim = sim_init
            else:
                sim = match_sim(Cp1, Cp2, sort=True, metric='cosine', norm_proc=['unit', 'center', 'unit'])
                sim_init = sim
                # sim = vecmap_unsup(Cp1, Cp2)

        if args.csls:
            sim = most_diff_match(sim, 10)
        inds1, inds2, evalsim = match(sim, args.match)

        if evalf is not None:
            evalf(evalsim)
        if end:
            break
        uniqf2 = uniqb1 = len(inds1)

        Cp1f = Cp1[:, inds1]
        Cp2f = Cp2[:, inds2]

        embeddings.normalize(Cp1f, normproc)
        embeddings.normalize(Cp2f, normproc)
        # maybe these matches
        sim = Cp1f @ Cp2f.T
        # X = torch.from_numpy(Cp1f)
        # Y = torch.from_numpy(Cp2f)
        # sim = -torch.cdist(X, Y, p=2).numpy()

        objf = np.mean(np.max(sim, axis=1))
        objb = np.mean(np.max(sim, axis=0))
        objective = 0.5 * (objf + objb)

        if objective > best_objective:
            last_improvement = it
            if it > 0: # the initial round use a different matrix and should not be compared 
                best_objective = objective
        print(f'objective {it} \t{objf:.5f} \t{objective:.5f} \t {best_objective:.5f} \t {uniqf2} \t {uniqb1}')

    return inds1, inds2, sim

def coocmapl1(Cp1: np.ndarray, Cp2: np.ndarray, args, normproc=['unit'], sim_init=None, evalf=None):
    """
    duplicated code using cdistance from torch, mainly to test l1 distance
    """
    best_objective = float('-inf')
    last_improvement = 0
    end = False
    inds1, inds2 = 0, 0
    simd = 0
    for it in range(args.maxiter):
        if it - last_improvement > args.stochastic_interval:
            end = True

        if it == 0:
            if sim_init is not None:
                sim = sim_init
            else:
                sim = match_sim(Cp1, Cp2, sort=True, metric='cosine', norm_proc=['unit', 'center', 'unit'])
                sim_init = sim
                # sim = vecmap_unsup(Cp1, Cp2)

        if args.csls:
            sim = most_diff_match(sim, 10)
        inds1, inds2, evalsim = match(sim, args.match)

        if evalf is not None:
            evalf(evalsim)
        if end:
            break
        uniqf2 = uniqb1 = len(inds1)

        Cp1f = Cp1[:, inds1]
        Cp2f = Cp2[:, inds2]

        embeddings.normalize(Cp1f, normproc)
        embeddings.normalize(Cp2f, normproc)
        # maybe these matches
        # sim = Cp1f @ Cp2f.T
        import torch
        if torch.cuda.is_available():
            X = torch.from_numpy(Cp1f).cuda()
            Y = torch.from_numpy(Cp2f).cuda()
            sim = -torch.cdist(X, Y, p=1).cpu().numpy()
        else:
            X = torch.from_numpy(Cp1f)
            Y = torch.from_numpy(Cp2f)
            sim = -torch.cdist(X, Y, p=1).numpy()

        # this is only approximately a greedy method, as this objective is not guaranteed to increase 
        objf = np.mean(np.max(sim, axis=1))
        objb = np.mean(np.max(sim, axis=0))
        objective = 0.5 * (objf + objb)

        if objective > best_objective:
            last_improvement = it
            if it > 0: # the initial round use a different matrix and should not be compared 
                best_objective = objective
        print(f'objective {it} \t{objf:.5f} \t{objective:.5f} \t {best_objective:.5f} \t {uniqf2} \t {uniqb1}')

    return inds1, inds2, sim

def svd_power(X, beta=1, drop=None, dim=None, symmetric=False):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    print('np.power(s)',  np.power(s, 1).sum())
    if dim is not None: 
        # s = np.sqrt(np.maximum(0, s**2 - s[dim]**2))
        # s = np.maximum(0, s - s[dim])
        s[dim:]=0
    print('np.power(s_dim)',  np.power(s, 1).sum())
    if dim is not None: 
        s = np.power(s, beta)
    if drop is not None:
        if isinstance(drop, np.ndarray):
            s[list(drop)] = 0
        elif isinstance(drop, int):
            s[:drop] = 0
    print('np.power(s_drop)',  np.power(s, 1).sum())
    if symmetric:
        res = (u * s) @ u.T
    else:
        res = (u * s) @ vt 
    norm = np.linalg.norm(res - X, ord='fro')
    normX = np.linalg.norm(X, ord='fro')
    print(f'diff {norm:.2e} / {normX:.2e}')
    return res

def sim_vecs(Co, dim, alpha=0.5, beta=1):
    maxdim = min(Co.shape[1], 10000)
    Co = Co[:, :maxdim]
    u, s, _ = np.linalg.svd(np.power(Co, alpha), full_matrices=False)
    u = u[:, :dim]*np.power(s[:dim], beta)
    return u

### matching methods ###
def greedy_match(sim0, iters=10):
    sim = sim0.copy()
    for i in range(iters):
        # if sim is n by m, am1 is size m, am0 is size n
        am1 = np.nanargmax(sim, axis=0)
        am0 = np.nanargmax(sim, axis=1)
        bi0 = am0[am1] == np.arange(sim.shape[1])
        bi1 = am1[am0] == np.arange(sim.shape[0])
        assert bi0.sum() == bi1.sum()
        bimatches = bi0.sum()
        uniques = len(np.unique(am0)), len(np.unique(am1))
        hubs = np.mean([c for _, c in Counter(am0).most_common(3)])
        value = np.take_along_axis(sim0, am1[:, None], axis=1).mean()
        stats = {'bimatches': bimatches, 'uniques': uniques, 'hubs': hubs, 'value': value}
        print(stats)
        if bimatches > 0.98 * min(*sim.shape):
            break

        for i in range(sim.shape[0]):
            if bi1[i]:
                sim[i] = float('nan') 
                sim[:, am0[i]] = float('nan')
                sim[i, am0[i]] = float('inf')
    return np.arange(sim.shape[1])[bi0], am0[bi0], sim

def most_diff_match(sim0, k):
    sim = sim0.copy()
    top0 = -np.partition(-sim, kth=k, axis=0)
    top1 = -np.partition(-sim, kth=k, axis=1)
    mean0 = top0[:k, :].mean(axis=0, keepdims=True)
    mean1 = top1[:, :k].mean(axis=1, keepdims=True)
    return sim - 0.5*(mean0 + mean1)

def forward_backward_match(sim):
    indsf2 = np.argmax(sim, axis=1)
    indsb1 = np.argmax(sim, axis=0)
    indsb2 = np.arange(sim.shape[1])
    indsf1 = np.arange(sim.shape[0])
    inds1 = np.concatenate((indsf1, indsb1))
    inds2 = np.concatenate((indsf2, indsb2))

    hubsf = Counter(indsf2).most_common(3)
    hubsb = Counter(indsb1).most_common(3)
    print('hubs', hubsf, hubsb)
    
    return inds1, inds2, sim

def match(sim, method):
    if method == 'vecmap':
        return forward_backward_match(sim)
    elif method == 'coocmap':
        return greedy_match(sim, iters=10)


### clipping ###
def clipthres(A, p1, p2):
    R1 = np.percentile(A, p1, axis=1, keepdims=True)
    r = np.percentile(R1, p2)
    print('percent greater \t', np.sum((A > r) * 1) / A.size)
    return r

def clipBoth(A, r1, r2):
    ub = clipthres(A, r1, r2) 
    lb = clipthres(A, 100-r1, 100-r2)
    print('clipped', lb, ub)
    return lb, ub

def clip(A, r1=99, r2=99):
    lb, ub = clipBoth(A, r1, r2)
    A[A < lb] = lb
    A[A > ub] = ub

