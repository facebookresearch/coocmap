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
    size1=50,
    # size2=20, skip2=10,
    symmetric=1,
    width=5,
    # vectorization fasttext sim_svd count
    vectorize='fasttext',
    dim=300,
    # tokenizer WordLevel, BPE
    tokentype='WordLevel',
    vocab_size=5000,
    limit_alphabet=100,
    min_frequency=5,
    # experiment
    supervision='unsupervised',
    label='iter100',
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
    threshold = 1e-4
    stochastic_initial = 1
    stochastic_multiplier = 2
    maxswaps = 100
    maxiter = 50
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

def record(type, sim):
    print('recording', type)
    simd = match.most_diff_match(sim, 10)
    df, stats = evaluation.report_sim(simd, d1.tokenizer, d2.tokenizer, dictpath)
    info = stats
    info.update({'id': run.id, 'method_type': type})
    for k, v in cfg.items():
        if k in info: print(f'Warning: {k} already exist')
        info[k] = v
    
    rows.append(info)
    wandb.log({'table': wandb.Table(dataframe=pd.DataFrame.from_records(rows))})
    wandb.log({'basicinfo': info})
    print(info)
    df.to_csv(os.path.join(dumpdir, f'{type}.csv'))


def experiment(dim):
    namemap = {'vecmap': 'bidir', 'coocmap': 'greedy'}
    name = namemap[args.match]
    label = f'-{name}-csls' if args.csls else f'{name}'

    print('original dim', d1.Co.shape)
    d1ft = d1.vecs[cfg.vectorize]
    d2ft = d2.vecs[cfg.vectorize]
    normproc = ['unit', 'center', 'unit']
    embeddings.normalize(d1ft, normproc)
    embeddings.normalize(d2ft, normproc)
    _, _, sim = match.vecmap(d1ft, d2ft, args, evalf=evalf)
    record('vecmap-fasttext' + label, sim)
    def sqrt_sim(x):
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return (u*s).dot(u.T)
    
    A1f = sqrt_sim(d1ft)
    A2f = sqrt_sim(d2ft)
    normproc = ['unit', 'center', 'unit']
    embeddings.normalize(A1f, normproc)
    embeddings.normalize(A2f, normproc)

    normproc1 = ['unit']
    _, _, simscoocmap = match.coocmapt(A1f, A2f, args, normproc=normproc1, sim_init=None, evalf=evalf)
    record(f'coocmap-{cfg.vectorize}-sqrt' + label, simscoocmap)

    A1 = np.sqrt(d1.Co)
    A2 = np.sqrt(d2.Co)
    dn1 = match.sim_vecs(A1, dim, alpha=1)
    dn2 = match.sim_vecs(A2, dim, alpha=1)
    # dn1 = np.array(d1.vec)
    # dn2 = np.array(d2.vec)
    embeddings.normalize(dn1, normproc)
    embeddings.normalize(dn2, normproc)
    
    _, _, sim =  match.vecmap(dn1, dn2, args, evalf=evalf)
    record('vecmap-raw' + label, sim)

    ###### coocmap ######

    embeddings.normalize(A1, normproc)
    embeddings.normalize(A2, normproc)

    normproc1 = ['unit']
    _, _, simscoocmap = match.coocmapt(A1, A2, args, normproc=normproc1, sim_init=None, evalf=evalf)
    record('coocmap' + label, simscoocmap)

    # what if I get the correspondence analysis vectors here??

    # sim0 = match.match_sim(A1, A2, sort=True, metric='cosine', norm_proc=['unit', 'center', 'unit'])
    normproc1 = ['unit']
    dictinit = dict_init_binary()
    _, _, simdictinit = match.coocmapt(A1, A2, args, normproc=normproc1, sim_init=dictinit, evalf=evalf)
    record('dict-init' + label, simdictinit)

args.match = 'vecmap'
args.csls = True
experiment(cfg.dim)
args.csls = False
experiment(cfg.dim)

# args.match = 'coocmap'
# args.csls = True
# experiment(cfg.dim)
# args.csls = False
# experiment(cfg.dim)

