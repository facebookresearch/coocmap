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
# os.environ["WANDB_MODE"] = "offline"

defaults = dict(
    # data
    lan1='enwikishuf',
    lan2='dewikishuf',
    eval='en-de',
    size1=300,
    # size2=20, skip2=10,
    symmetric=1,
    width=5,
    # vectorization fasttext sim_svd count
    vectorize='trunc',
    dim=300,
    # tokenizer WordLevel, BPE
    tokentype='WordLevel',
    vocab_size=500,
    limit_alphabet=100,
    min_frequency=5,
    # experiment
    supervision='unsupervised',
    label='glove',
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
    stochastic_interval = 10
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

def dict_init_binary(tiebreak=1e-3):
    inds = evaluation.dict_to_inds(dictpath, d1.tokenizer, d2.tokenizer, full=False)
    sim = tiebreak * np.random.rand(d1.Co.shape[0], d2.Co.shape[0])
    for i in range(len(inds[0])):
        sim[inds[0][i], inds[1][i]] = 1
    sim[0, 0] = 1
    return sim

rows = []

def experiment(drop=20, dim=300):
    print('original dim', d1.Co.shape)
    def record(type, sim):
        print(type)
        # plt.figure()
        # plt.imshow(sims[type])
        simd = match.most_diff_match(sim, 10)
        # inds1, inds2, sim_greed = match.greedy_match(simd, iters=5)
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


    def standard_normalize(normproc, name):
        A1 = np.array(1.0*d1.Co)
        A2 = np.array(1.0*d2.Co)
        embeddings.normalize(A1, normproc)
        embeddings.normalize(A2, normproc)
        _, _, simscoocmap = match.coocmapt(A1, A2, args, normproc=normproc1, sim_init=None, evalf=evalf)
        record(name, simscoocmap)
    
    # standard_normalize(['log'] + normproc, 'log')
    standard_normalize(['log1p'] + normproc, 'log1p')

    def levy2014():
        standard_normalize(['levy2014', 'unit'], 'levy2014-l2')
        standard_normalize(['levy2014'] + normproc, 'levy2014-normalize')

        standard_normalize(['levy2014_k5', 'unit'], 'levy2014_k5-l2')
        standard_normalize(['levy2014_k5'] + normproc, 'levy2014_k5-normalize')

        A1 = np.array(1.0*d1.Co)
        A2 = np.array(1.0*d2.Co)
        embeddings.normalize(A1, ['levy2014', 'unitL1'])
        embeddings.normalize(A2, ['levy2014', 'unitL1'])
        _, _, simscoocmap = match.coocmapl1(A1, A2, args, normproc=['unitL1'], sim_init=None, evalf=evalf)
        record('levy2014-l1', simscoocmap)

        A1 = np.array(1.0*d1.Co)
        A2 = np.array(1.0*d2.Co)
        embeddings.normalize(A1, ['levy2014_k5', 'unitL1'])
        embeddings.normalize(A2, ['levy2014_k5', 'unitL1'])
        _, _, simscoocmap = match.coocmapl1(A1, A2, args, normproc=['unitL1'], sim_init=None, evalf=evalf)
        record('levy2014_k5-l1', simscoocmap)
    
    def glove():
        standard_normalize(['glove', 'unit'], 'glove-l2')
        standard_normalize(['glove'] + normproc, 'glove-normalize')

        A1 = np.array(1.0*d1.Co)
        A2 = np.array(1.0*d2.Co)
        embeddings.normalize(A1, ['glove', 'unitL1'])
        embeddings.normalize(A2, ['glove', 'unitL1'])
        _, _, simscoocmap = match.coocmapl1(A1, A2, args, normproc=['unitL1'], sim_init=None, evalf=evalf)
        record('glove-l1', simscoocmap)
    glove()


    # A1 = np.sqrt(d1.Co)
    # A2 = np.sqrt(d2.Co)
    # embeddings.normalize(A1, normproc)
    # embeddings.normalize(A2, normproc)
    # _, _, simscoocmap = match.coocmapt(A1, A2, args, normproc=normproc1, sim_init=None, evalf=evalf)
    # record('coocmap', simscoocmap)

    # A1c = np.array(A1)
    # A2c = np.array(A2)
    # match.clip(A1c, r1=99, r2=99)
    # match.clip(A2c, r1=99, r2=99)
    # _, _, simscoocmap = match.coocmapt(A1c, A2c, args, normproc=normproc1, sim_init=None, evalf=evalf)
    # record('coocmap-clip', simscoocmap)

    # A1f = match.svd_power(A1, beta=1, drop=drop, dim=None)
    # A2f = match.svd_power(A2, beta=1, drop=drop, dim=None)
    # normproc = ['unit', 'center', 'unit']
    # embeddings.normalize(A1f, normproc)
    # embeddings.normalize(A2f, normproc)
    # match.clip(A1f, r1=99, r2=99)
    # match.clip(A2f, r1=99, r2=99)

    # # dropinit = match.most_diff_match(simscoocmap, 10)
    # dropinit = simscoocmap
    # _, _, sim = match.coocmapt(A1f, A2f, args, normproc=normproc1, sim_init=dropinit, evalf=evalf)
    # record('coocmap-drop', sim)
    # clipinit = sim

    def rapp1995(name, init):
        alpha = 1.0
        A1f = np.power(d1.Co, alpha)
        A2f = np.power(d2.Co, alpha)
        norm = ['pmi', 'unitL1']
        embeddings.normalize(A1f, norm)
        embeddings.normalize(A2f, norm)
        _, _, simscoocmap = match.coocmapl1(A1f, A2f, args, normproc=['unitL1'], sim_init=init, evalf=evalf)
        record(name, simscoocmap)
    # rapp1995('rapp1995', None)
    # rapp1995('rapp1995-init', clipinit)

    def fung1997(name, init):
        alpha = 1.0
        A1f = np.power(d1.Co, alpha)
        A2f = np.power(d2.Co, alpha)
        norm = ['fung1997', 'unitL1']
        embeddings.normalize(A1f, norm)
        embeddings.normalize(A2f, norm)
        _, _, simscoocmap = match.coocmapl1(A1f, A2f, args, normproc=['unitL1'], sim_init=init, evalf=evalf)
        record(name, simscoocmap)
    # fung1997('fung1997-l1', None)
    # fung1997('fung1997-l1-init', clipinit)

    # normproc1 = ['unit']
    # dictinit = dict_init_binary()
    # _, _, simdictinit = match.coocmapt(A1f, A2f, args, normproc=normproc1, sim_init=dictinit, evalf=evalf)
    # record('dict-init-drop', simdictinit)

# generate a simple grid enumeration
from itertools import product
drops = [20]
dims = [100]
grid_plan = product(drops, dims)

for drop, dim in grid_plan:
    if drop >= dim: continue
    experiment(drop, dim)

# method = VecMap(d1.vecpath, d2.vecpath, dictpath, wandb.run.dir, cfg)
# method.run()
# res = method.eval(dictpath)
# # write the predictions
# wandb.log({'accuracy': res['accuracy'], 'coverage': res['coverage']})