import os
import subprocess
from dataclasses import dataclass
import lzma
import wandb
import argparse
import shutil 
import pandas as pd
import numpy as np

import data
import match
import evaluation
import embeddings
# from baselines import VecMap

os.environ['WANDB_IGNORE_GLOBS'] = 'lan1/*,lan2/*'
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
defaults = dict(
    # data
    lan1='enwikishuf',
    lan2='eswikishuf',
    eval='en-es',
    size1=10,
    # size2=20, skip2=10,
    symmetric=1,
    width=5,
    # vectorization fasttext sim_svd count
    vectorize='trunc',
    dim=300,
    # tokenizer WordLevel, BPE
    tokentype='WordLevel',
    vocab_size=100,
    limit_alphabet=100,
    min_frequency=1,
    # experiment
    supervision='basic-init',
    label='none',
)

run = wandb.init(config=defaults, project='data efficiency')
base1 = os.path.join(wandb.run.dir, 'lan1')
base2 = os.path.join(wandb.run.dir, 'lan2')
os.makedirs(base1)
os.makedirs(base2)
cfg = wandb.config

def make_sized(lan, sizemb, pout, skipmb=0):
    corpus = data.get_data(lan)
    text = corpus.headmb(lan, skipmb+sizemb)
    with open(pout, 'wt', encoding='utf-8') as fout:
        fout.write(text[int(skipmb*1e6):])

p1 = os.path.join(base1, 'c.txt')
p2 = os.path.join(base2, 'c.txt')

make_sized(cfg.lan1, cfg.size1, p1)

size2 = cfg.size1 if cfg.symmetric == 1 else cfg.size2
skip2 = cfg.size1 if cfg.lan1 == cfg.lan2 else 0
make_sized(cfg.lan2, size2, p2, skipmb=skip2)

##### often fails

d1 = data.Corpus(p1, base1,
                 tokentype=cfg.tokentype, vocab_size=cfg.vocab_size, limit_alphabet=cfg.limit_alphabet, min_frequency=cfg.min_frequency,
                 vectorize=cfg.vectorize, width=cfg.width, dim=cfg.dim)
d2 = data.Corpus(p2, base2,
                 tokentype=cfg.tokentype, vocab_size=cfg.vocab_size, limit_alphabet=cfg.limit_alphabet, min_frequency=cfg.min_frequency,
                 vectorize=cfg.vectorize, width=cfg.width, dim=cfg.dim)

def get_evaldict():
    lan1s, lan2s = cfg.eval.split('-')
    eval = data.MUSEEval()
    dictpath = os.path.join(wandb.run.dir, 'eval_id.dict')
    with open(dictpath, 'wt', encoding='utf-8', errors='surrogateescape') as f:
        v1 = d1.tokenizer.get_vocab()
        v2 = d2.tokenizer.get_vocab()
        intersection = set(v1.keys()).intersection(v2.keys())
        for w in intersection:
            f.write(f'{w}\t{w}\n')
    # dictid = dictpath

    if lan1s != lan2s:
        dictpath = os.path.join(wandb.run.dir, 'eval_dict.dict')
        lanpath = eval.eval_path(f'{lan1s}-{lan2s}', type='full')
        shutil.copyfile(lanpath, dictpath)

    return dictpath

dictpath = get_evaldict()

@dataclass
class SearchArgs:
    stochastic_interval = 20
    stochastic_add = 1e-1
    stochastic_multiplier = 2
    threshold = 1e-4
    stochastic_initial = 1
    maxswaps = 100
    maxiter = 100
    eta = 1
    #
    method = 'orthogonal' # or orthogonal or lstsq
    match = 'vecmap'
    csls = True
args = SearchArgs()

dumpdir = os.path.join(wandb.run.dir, 'dump')
os.makedirs(dumpdir, exist_ok=True)

def evalf(sim):
    # simf = match.most_diff_match(sim, k=3)
    f, stats = evaluation.report_sim(sim, d1.tokenizer, d2.tokenizer, dictpath)
    print(stats)

def dict_init_binary():
    inds = evaluation.dict_to_inds(dictpath, d1.tokenizer, d2.tokenizer, full=False)
    sim = np.zeros((d1.Co.shape[0], d2.Co.shape[0]))
    for i in range(len(inds[0])):
        sim[inds[0][i], inds[1][i]] = 1
    return sim

rows = []
def experiment(drop=20, dim=300, r1=1, r2=1):
    print('original dim', d1.Co.shape)
    def record(type, sim):
        print(type)
        # plt.figure()
        # plt.imshow(sims[type])
        simd = match.most_diff_match(sim, 10)
        df, stats = evaluation.report_sim(simd, d1.tokenizer, d2.tokenizer, dictpath)
        info = stats
        info.update({'id': run.id, 'drop': drop, 'dim_p': dim, 'method_type': type})
        for k, v in cfg.items():
            if k in info: print(f'Warning: {k} already exist')
            info[k] = v
        
        rows.append(info)
        wandb.log({'table': wandb.Table(dataframe=pd.DataFrame.from_records(rows))})
        wandb.log({'basicinfo': info})
        print(info)
        df.to_csv(os.path.join(dumpdir, f'{type}-{drop}-{dim}.csv'))

    normproc1 = ['unit']
    normproc = ['unit', 'center', 'unit']

    A1 = np.sqrt(d1.Co)
    A2 = np.sqrt(d2.Co)
    embeddings.normalize(A1, normproc)
    embeddings.normalize(A2, normproc)

    if cfg.supervision == 'common-init': 
        coocinit = match.match_sim(A1, A2, sort=True, metric='cosine', norm_proc=['unit', 'center', 'unit'])
    elif cfg.supervision == 'clip-init':
        A1c = np.array(A1)
        A2c = np.array(A2)
        match.clip(A1c, r1=99, r2=99)
        match.clip(A2c, r1=99, r2=99) 
        coocinit = match.match_sim(A1c, A2c, sort=True, metric='cosine', norm_proc=['unit', 'center', 'unit'])
    else:
        coocinit = None

    # d1ft = d1.vecs[cfg.vectorize]
    # d2ft = d2.vecs[cfg.vectorize]
    # embeddings.normalize(d1ft, normproc)
    # embeddings.normalize(d2ft, normproc)
    # _, _, sim = match.vecmap(d1ft, d2ft, args, evalf=evalf, sim_init=coocinit)
    # record(f'vecmap-{cfg.vectorize}', sim)

    def coocmapvec():
        def sqrt_sim(x):
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            return (u*s).dot(u.T)
        
        A1f = sqrt_sim(d1ft)
        A2f = sqrt_sim(d2ft)
        embeddings.normalize(A1f, normproc)
        embeddings.normalize(A2f, normproc)
        _, _, simscoocmap = match.coocmapt(A1f, A2f, args, normproc=normproc1, sim_init=coocinit, evalf=evalf)
        record(f'coocmap-{cfg.vectorize}', simscoocmap)

        A1c = np.array(A1f)
        A2c = np.array(A2f)
        match.clip(A1c, r1=99, r2=99)
        match.clip(A2c, r1=99, r2=99)
        _, _, simscoocmap = match.coocmapt(A1c, A2c, args, normproc=normproc1, sim_init=coocinit, evalf=evalf)
        record(f'coocmap-{cfg.vectorize}-clip', simscoocmap)
        initdrop = simscoocmap

        A1d = match.svd_power(A1f, beta=1, drop=drop, dim=dim)
        A2d = match.svd_power(A2f, beta=1, drop=drop, dim=dim)
        embeddings.normalize(A1d, normproc)
        embeddings.normalize(A2d, normproc)
        match.clip(A1d, r1=99, r2=99)
        match.clip(A2d, r1=99, r2=99)
        _, _, simscoocmap = match.coocmapt(A1d, A2d, args, normproc=normproc1, sim_init=initdrop, evalf=evalf)
        record(f'coocmap-{cfg.vectorize}-drop', simscoocmap)
    # coocmapvec()

    # what if I get the correspondence analysis vectors here??
    # sim0 = match.match_sim(A1, A2, sort=True, metric='cosine', norm_proc=['unit', 'center', 'unit'])
    A1f = match.svd_power(A1, beta=1, drop=None, dim=dim)
    A2f = match.svd_power(A2, beta=1, drop=None, dim=dim)
    embeddings.normalize(A1f, normproc)
    embeddings.normalize(A2f, normproc)
    # _, _, simscoocmap = match.coocmapt(A1f, A2f, args, normproc=normproc1, sim_init=coocinit, evalf=evalf)
    # record('coocmap', simscoocmap)


    A1c = np.array(A1f)
    A2c = np.array(A2f)
    match.clip(A1c, r1=100-r1, r2=100-r2)
    match.clip(A2c, r1=100-r1, r2=100-r2)
    _, _, simscoocmap = match.coocmapt(A1c, A2c, args, normproc=normproc1, sim_init=coocinit, evalf=evalf)
    record(f'coocmap-clip-{r1:.1f}-{r2:.1f}', simscoocmap)
    dropinit = simscoocmap

    A1f = match.svd_power(A1, beta=1, drop=drop, dim=dim)
    A2f = match.svd_power(A2, beta=1, drop=drop, dim=dim)
    embeddings.normalize(A1f, normproc)
    embeddings.normalize(A2f, normproc)
    match.clip(A1f, r1=100-r1, r2=100-r2)
    match.clip(A2f, r1=100-r1, r2=100-r2)
    # dropinit = match.most_diff_match(simscoocmap, 10)
    _, _, sim = match.coocmapt(A1f, A2f, args, normproc=normproc1, sim_init=dropinit, evalf=evalf)
    record(f'coocmap-drop-{r1:.1f}-{r2:.1f}', sim)


    A1f = match.svd_power(A1c, beta=1, drop=drop, dim=dim)
    A2f = match.svd_power(A2c, beta=1, drop=drop, dim=dim)
    embeddings.normalize(A1f, normproc)
    embeddings.normalize(A2f, normproc)
    match.clip(A1f, r1=100-r1, r2=100-r2)
    match.clip(A2f, r1=100-r1, r2=100-r2)
    _, _, sim = match.coocmapt(A1f, A2f, args, normproc=normproc1, sim_init=dropinit, evalf=evalf)
    record(f'coocmap-clip-drop-{r1:.1f}-{r2:.1f}', sim)

# generate a simple grid enumeration
r1 = [0.5, 1, 2, 5]
r2 = [0.5, 1, 2, 5]
from itertools import product
grid_plan = list(product(r1, r2))
print(grid_plan)
np.random.shuffle(grid_plan)
for r1, r2 in grid_plan:
    drop = np.ceil(min(20, int(cfg.dim) * 20/400)) # 400 -> 20
    experiment(drop, int(cfg.dim), r1, r2)

# method = VecMap(d1.vecpath, d2.vecpath, dictpath, wandb.run.dir, cfg)
# method.run()
# res = method.eval(dictpath)
# # write the predictions
# wandb.log({'accuracy': res['accuracy'], 'coverage': res['coverage']})