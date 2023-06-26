# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from dataclasses import dataclass
import wandb
import shutil 
import pandas as pd
import numpy as np

import data
import match
import evaluation
import embeddings

# experimental parameters
defaults = dict(
    lan1='./europarl-v7.hu-en.en',
    lan2='./europarl-v7.hu-en.hu',
    eval='en-hu',
    size1=20,

    width=5,
    symmetric=1,
    vectorize='trunc',  # fasttext sim_svd trunc word2vec
    dim=300,
    tokentype='WordLevel',  # tokenizer WordLevel, BPE
    vocab_size=5000,
    limit_alphabet=100,
    min_frequency=5,
    supervision='unsupervised',
    label='none',
)

os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
os.environ['WANDB_IGNORE_GLOBS'] = 'lan1/*,lan2/*'
os.environ["WANDB_MODE"] = "offline" # switch to "online" to use wandb cloud sync

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
        print('vocab has overlap of length', len(intersection))
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
    stochastic_interval = 5
    stochastic_initial = 1
    stochastic_multiplier = 2
    threshold = 1e-4
    maxiter = 100
    eta = 1
    method = 'orthogonal' # or orthogonal or lstsq
    match = 'vecmap'
    csls = True
args = SearchArgs()

dumpdir = os.path.join(wandb.run.dir, 'dump')
os.makedirs(dumpdir, exist_ok=True)

def evalf(sim):
    # simf = match.most_diff_match(sim, k=10)
    f, stats = evaluation.report_sim(sim, d1.tokenizer, d2.tokenizer, dictpath)
    print(stats)

def dict_init_binary(tiebreak=1e-3):
    inds = evaluation.dict_to_inds(dictpath, d1.tokenizer, d2.tokenizer, full=False)
    sim = tiebreak * np.random.rand(d1.Co.shape[0], d2.Co.shape[0])
    for i in range(len(inds[0])):
        sim[inds[0][i], inds[1][i]] = 1
    sim[0, 0] = 1
    return sim

rows = []

def experiment(drop=20, dim=300, r1=99, r2=99):
    def record(type, sim):
        print(type)
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

    normproc = ['unit', 'center', 'unit']
    normproc1 = ['unit']

    def f(Co):
        X = np.sqrt(Co)
        embeddings.normalize(X, normproc)
        return X
    X, Z = f(d1.Co), f(d2.Co)
    _, _, simscoocmap = match.coocmapt(X, Z, args, normproc=normproc1, sim_init=None, evalf=evalf)
    record('coocmap', simscoocmap)

    def clip_drop():
        def f(Co):
            X = np.sqrt(Co)
            embeddings.normalize(X, normproc)
            match.clip(X, r1=r1, r2=r2)
            return X
        X, Z = f(d1.Co), f(d2.Co)
        _, _, simscoocmap = match.coocmapt(X, Z, args, normproc=normproc1, sim_init=None, evalf=evalf)
        record('coocmap-clip', simscoocmap)

        dropinit = simscoocmap

        def f(Co):
            X = np.sqrt(Co)
            embeddings.normalize(X, normproc)
            X = match.svd_power(X, beta=1, drop=drop, dim=None)
            embeddings.normalize(X, normproc)
            match.clip(X, r1=r1, r2=r2)
            return X
        X, Z = f(d1.Co), f(d2.Co)
        _, _, simscoocmap = match.coocmapt(X, Z, args, normproc=normproc1, sim_init=dropinit, evalf=evalf)
        record('coocmap-drop', simscoocmap)
    # clip_drop() # run clip and drop as well

experiment(drop=20, dim=300)
