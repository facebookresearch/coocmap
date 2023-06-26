# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional
import collections

import numpy as np
import pandas as pd

from tokenizers import Tokenizer

# faithfully recreate the protocol of vecmap with minimal code modifications
def vecmap_evaluate(sim: np.ndarray, tokenizer1: Tokenizer, tokenizer2: Tokenizer, refpath: str):
    # https://github.com/artetxem/vecmap/blob/master/map_embeddings.py#L225
    # precision only, count oovs
    with open(refpath, encoding='utf-8', errors='surrogateescape') as f:
        validation = collections.defaultdict(set)
        oov = set()
        vocab = set()
        for line in f:
            try:
                src, trg = line.split()
            except ValueError:
                continue
            try:
                src_ind = tokenizer1.token_to_id(src)
                trg_ind = tokenizer2.token_to_id(trg)
                if src_ind is None or trg_ind is None:
                    raise KeyError
                if src_ind >= sim.shape[0] or trg_ind >= sim.shape[1]:
                    raise KeyError

                validation[src_ind].add(trg_ind)
                vocab.add(src)
            except KeyError:
                oov.add(src)
        oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
        validation_coverage = len(validation) / (len(validation) + len(oov))

    # https://github.com/artetxem/vecmap/blob/master/map_embeddings.py#L383
    src = list(validation.keys())
    # xw[src].dot(zw.T, out=simval)
    srct = [s for s in src if s < sim.shape[0]]
    simval = sim[srct]
    nn = np.nanargmax(simval, axis=1)
    accuracy = np.mean([1 if nn[i] in validation[src[i]] else 0 for i in range(len(src))])
    similarity = np.mean([max([simval[i, j].tolist() for j in validation[src[i]]]) for i in range(len(src))])
    return {'accuracy': accuracy, 'similarity': similarity, 'coverage': validation_coverage}

def get_refdict(refpath):
    with open(refpath, encoding='utf-8', errors='surrogateescape') as f:
        val = collections.defaultdict(set)
        for line in f:
            try:
                src, trg = line.split()
            except ValueError:
                continue
            val[src].add(trg)
        return val

def report_sim(sim: np.ndarray, tokenizer1: Tokenizer, tokenizer2: Tokenizer, refpath: Optional[str]):
    # ind_src = np.arange(sim.shape[0])
    kth = range(3)
    ind_tgt = np.argpartition(-sim, kth, axis=1)

    res = []
    maxes = []
    stats = {}
    if refpath is not None:
        refdict = get_refdict(refpath)
        # keys: accuracy, coverage, similarity
        vecmapres = vecmap_evaluate(sim, tokenizer1, tokenizer2, refpath)
        stats = vecmapres
    else:
        refdict = collections.defaultdict(set)

    for i in range(sim.shape[0]):
        char = tokenizer1.id_to_token(i)
        pred = tokenizer2.id_to_token(ind_tgt[i][0])
        preds = ' '.join(tokenizer2.id_to_token(j) for j in ind_tgt[i][kth])
        gap = sim[i][ind_tgt[i][0]] - sim[i][ind_tgt[i][1]]
        maxes.append(sim[i][ind_tgt[i][0]])
        res.append({
            'char': char,
            # 'id': i,
            'pred': pred,
            'preds': preds,
            'eq': char == pred,
            # 'gap': gap,
            # 'max': maxes[i],
            'correct': pred in refdict[char],
            'refs': ' '.join(refdict[char])
        })
    # print(res)
    df = pd.DataFrame.from_records(res)
    neq = len(df[df['char'] == df['pred']])
    ncorrect = len(df[df['correct']==True])

    stats['nidentical'] = neq
    stats['mean_max'] = np.mean(maxes)
    stats['ncorrrect'] = ncorrect
    # print(stats)

    return df, stats


def _dict_to_inds(refpath, tok1, tok2, full=False):
    refdict = get_refdict(refpath)
    for src, trgs in refdict.items():
        src_ind = tok1.token_to_id(src)
        if src_ind is None:
            continue
        trg_inds = [tok2.token_to_id(trg) for trg in trgs]
        trg_inds = [trg_ind for trg_ind in trg_inds if trg_ind is not None]
        if full:
            for trg_ind in trg_inds:
                yield src_ind, trg_ind
        elif len(trg_inds) > 0:
            trg_ind = trg_inds[0]
            yield src_ind, trg_ind

def dict_to_inds(refpath, tok1, tok2, full=False):
    return list(zip(*_dict_to_inds(refpath, tok1, tok2, full=full)))

def label_preds(preds, refpath: Optional[str]):
    # ind_src = np.arange(sim.shape[0])
    if refpath is not None:
        refdict = get_refdict(refpath)
        print('size of dictionary', len(refdict.keys()))

    res = []
    for w, v in preds:
        res.append(
            {
            'src': w,
            'trg': v,
            'correct': v in refdict[w],
            'wrong': w in refdict and v not in refdict[w],
            'identical': w == v,
            'refs': ' '.join(refdict[w]),
            }
        )
        ws.append(w)

    if len(ws) != len(set(ws)):
        print('WARNING: duplicate words exist in the predictions')

    # print(res)
    df = pd.DataFrame.from_records(res)
    def boolcount(prop):
        return len(df[df[prop]==True])
    nidentical = boolcount('identical') 
    ncorrect = boolcount('correct') 
    nwrong= boolcount('wrong') 
    accuracy = ncorrect / (ncorrect + nwrong)
    coverage = (ncorrect + nwrong) / len(refdict)
    noov = len(refdict) - (ncorrect + nwrong)
    stats = {'nidentical': nidentical, 'ncorrect': ncorrect, 'noov': noov, 'accuracy': accuracy, 'coverage': coverage}
    return df, stats