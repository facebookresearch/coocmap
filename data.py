# Copyright (c) Meta Platforms, Inc. and affiliates.
import itertools
import os
import sys
import subprocess
import time
# import lzma # needed for BUCC20Corpus

import numpy as np

from tokenizers import Token, Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Metaspace, Whitespace, WhitespaceSplit
from tokenizers.normalizers import Lowercase 

from fast import cooc_count

os.environ["TOKENIZERS_PARALLELISM"] = "true"
cachebase = os.path.expanduser('~/.cache/cooc/')

def full_path(lan):
    if lan.startswith('~/') or lan.startswith('./') or lan.startswith('/'):
        path = os.path.expanduser(lan)
        print('trying path', path)
    else:
        # relative path from cachebase
        path = os.path.expanduser(os.path.join(cachebase, lan))
        print('trying cache path', path)
    return path


def get_data(lan):
    if lan in EuroParl.lans:
        return EuroParl()
    if lan in WikiDump.lans:
        return WikiDump()
    if lan in NewsCrawl.lans:
        return NewsCrawl()

    # else just get raw file from cache base
    """
    wikidumps/zhwikishuf.jieba.txt: tokenized and to lower case
    news-crawl/news.2018.en.shuffled.deduped: en-news for a change
    """

    path = full_path(lan)
    if os.path.isfile(path):
        return HeadableData()
    else:
        raise Exception(f'No data for {lan} at {path}')
        

class HeadableData(object):
    cachedir = os.path.expanduser(os.path.join(cachebase))
    # works if you specify the path relative to the cachedir
    def headmb(self, lan, sizemb):
        size = int(sizemb * 1000000)
        lantxt = full_path(lan)
        f = open(lantxt, 'rt', encoding="utf-8")
        sizedtxt = f.read(size)
        return sizedtxt

class NewsCrawl(HeadableData):
    """
    Data from https://data.statmt.org/news-crawl/en/
    https://data.statmt.org/news-crawl/es/

    processing of this data was very simple, so just notes here
    wget https://data.statmt.org/news-crawl/en/news.2018.en.shuffled.deduped.gz
    wget https://data.statmt.org/news-crawl/es/news.2019.es.shuffled.deduped.gz
    wget https://data.statmt.org/news-crawl/hu/news.2019.hu.shuffled.deduped.gz
    wget https://data.statmt.org/news-crawl/hu/news.2020.hu.shuffled.deduped.gz
    wget https://data.statmt.org/news-crawl/hu/news.2021.hu.shuffled.deduped.gz
    cat news.*.hu.* > news.2019-2021.hu.shuffled.deduped

    This one removes around half the data
    cat news.2019-2021.hu.shuffled.deduped  | grep -v http | grep -v trackingCode > news.2019-2021.hu.shuffled.deduped.filtered
    gzip -d *
    """
    cachedir = os.path.expanduser(os.path.join(cachebase, 'news-crawl/'))
    lans = ['news.2018.en.shuffled.deduped', 'news.2019.es.shuffled.deduped', 'news.2019-2021.hu.shuffled.deduped', 'news.2019-2021.hu.shuffled.deduped.filtered', 'news.2018-2019-2020-2022.hu.shuffled']
    def headmb(self, lan, sizemb):
        assert lan in self.lans, 'lan must be one of: ' + ', '.join(self.lans)
        size = int(sizemb * 1000000)
        lantxt = os.path.join(self.cachedir, f'{lan}')
        f = open(lantxt, 'rt', encoding="utf-8")
        sizedtxt = f.read(size)
        return sizedtxt


class EuroParl(HeadableData):
    cachedir = os.path.expanduser(os.path.join(cachebase, 'europarl/'))
    urls = {
        # 'fr-en': 'https://www.statmt.org/europarl/v7/fr-en.tgz',
        # 'es-en': 'https://www.statmt.org/europarl/v7/es-en.tgz',
        # 'de-en': 'https://www.statmt.org/europarl/v7/de-en.tgz',
        'fi-en': 'https://www.statmt.org/europarl/v7/fi-en.tgz',
        'hu-en': 'https://www.statmt.org/europarl/v7/hu-en.tgz',
    }
    lans_raw = [f'europarl-v7.{suf}' for suf in ['fr-en.fr', 'fr-en.en', 'es-en.es', 'es-en.en', 'de-en.de', 'hu-en.en', 'hu-en.hu', 'fi-en.fi', 'fi-en.en']]
    lansshuf = [f'{pref}.shuf' for pref in lans_raw]
    lans = lans_raw + lansshuf
    def __init__(self):
        cachedir = self.cachedir
        if not os.path.isdir(cachedir):
            print(f'Making dir {cachedir}', file=sys.stderr)
            os.makedirs(cachedir, exist_ok=True)

    def check_and_dl_all(self):
        for lan in self.urls:
            self.check_and_dl(lan)
        
        for l in self.lans_raw:
            if not os.path.isfile(os.path.join(self.cachedir, l)):
                tgzname = l.split('.')[1] + '.tgz'
                print(f'Extracting for {l}', file=sys.stderr)
                proc = subprocess.run(f'tar xzf {tgzname}', shell=True, cwd=self.cachedir)
            else:
                print(f'Already extracted for {l}', file=sys.stderr)
    
        for flan, fshuf in zip(self.lans_raw, self.lansshuf):
            if not os.path.isfile(os.path.join(self.cachedir, fshuf)):
                subprocess.run(f'cat {flan} | shuf > {fshuf}', shell=True, cwd=self.cachedir)

    def check_and_dl(self, lan):
        url = self.urls[lan]

        fname = url.split('/')[-1]
        outfile = os.path.join(self.cachedir, fname)

        if not os.path.isfile(outfile):
            print(f'Downloading {outfile}', file=sys.stderr)
            proc = subprocess.run(f'wget -nv {url} -O {outfile}', shell=True, cwd=self.cachedir)
        else:
            print(f'Already downloaded {outfile}', file=sys.stderr) 
    
    def headmb(self, lan, sizemb):
        assert lan in self.lans, 'lan must be one of: ' + ', '.join(self.lans)
        size = int(sizemb * 1000000)
        lantxt = os.path.join(self.cachedir, f'{lan}')
        f = open(lantxt, 'rt', encoding="utf-8")
        sizedtxt = f.read(size)
        return sizedtxt 
        
    # https://www.statmt.org/europarl/v7/es-en.tgz
    # wc: 2007723 52653110 346919801 europarl-v7.fr-en.fr
    # wc: 2007723 50330641 301523301 europarl-v7.fr-en.en

class WikiDump(HeadableData):
    """
    
    """
    urls = {
        'enwiki': [ 
            'https://dumps.wikimedia.org/enwiki/20230401/enwiki-20230401-pages-meta-current1.xml-p1p41242.bz2',
            'https://dumps.wikimedia.org/enwiki/20230401/enwiki-20230401-pages-meta-current2.xml-p41243p151573.bz2',
            'https://dumps.wikimedia.org/enwiki/20230401/enwiki-20230401-pages-meta-current3.xml-p151574p311329.bz2'
        ],
        'eswiki': [
            'https://dumps.wikimedia.org/eswiki/20230401/eswiki-20230401-pages-meta-current1.xml-p1p159400.bz2',
            'https://dumps.wikimedia.org/eswiki/20230401/eswiki-20230401-pages-meta-current2.xml-p159401p693323.bz2',
            'https://dumps.wikimedia.org/eswiki/20230401/eswiki-20230401-pages-meta-current3.xml-p693324p1897740.bz2'
        ],
        'zhwiki': [
            'https://dumps.wikimedia.org/zhwiki/20230401/zhwiki-20230401-pages-meta-current1.xml-p1p187712.bz2',
            'https://dumps.wikimedia.org/zhwiki/20230401/zhwiki-20230401-pages-meta-current2.xml-p187713p630160.bz2',
            'https://dumps.wikimedia.org/zhwiki/20230401/zhwiki-20230401-pages-meta-current3.xml-p630161p1389648.bz2',
            'https://dumps.wikimedia.org/zhwiki/20230401/zhwiki-20230401-pages-meta-current4.xml-p1389649p2889648.bz2',
            'https://dumps.wikimedia.org/zhwiki/20230401/zhwiki-20230401-pages-meta-current4.xml-p2889649p3391029.bz2',
            'https://dumps.wikimedia.org/zhwiki/20230401/zhwiki-20230401-pages-meta-current5.xml-p3391030p4891029.bz2'
        ],
        'frwiki': [
            'https://dumps.wikimedia.org/frwiki/20230401/frwiki-20230401-pages-meta-current1.xml-p1p306134.bz2',
            'https://dumps.wikimedia.org/frwiki/20230401/frwiki-20230401-pages-meta-current2.xml-p306135p1050822.bz2',
            'https://dumps.wikimedia.org/frwiki/20230401/frwiki-20230401-pages-meta-current3.xml-p1050823p2550822.bz2'
        ],
        'dewiki': [
            'https://dumps.wikimedia.org/dewiki/20230401/dewiki-20230401-pages-meta-current1.xml-p1p297012.bz2',
            'https://dumps.wikimedia.org/dewiki/20230401/dewiki-20230401-pages-meta-current2.xml-p297013p1262093.bz2',
            'https://dumps.wikimedia.org/dewiki/20230401/dewiki-20230401-pages-meta-current3.xml-p1262094p2762093.bz2'
        ]
    }
    lans = {'enwikishuf', 'eswikishuf', 'zhwikishuf', 'frwikishuf', 'dewikishuf'}
    cachedir = os.path.expanduser(os.path.join(cachebase, 'wikidumps/'))

    def __init__(self):
        cachedir = self.cachedir
        if not os.path.isdir(cachedir):
            print(f'Making dir {cachedir}', file=sys.stderr)
            os.makedirs(cachedir, exist_ok=True)
    
    def check_and_dl_all(self):
        for lan in self.urls:
            self.check_and_dl(lan)

    def check_and_dl(self, lan):
        landir = os.path.join(self.cachedir, lan)
        if not os.path.isdir(landir):
            os.makedirs(landir, exist_ok=True)

        urls = self.urls[lan]
        for partn, url in enumerate(urls):
            # get last part of the url
            fname = url.split('/')[-1]
            outfile = os.path.join(landir, fname)

            if not os.path.isfile(outfile):
                print(f'Downloading {outfile}', file=sys.stderr)
                proc = subprocess.Popen(['wget', '-nv', url, '-O', outfile])
                output, error = proc.communicate()
                print(output, file=sys.stderr)
                print(error, file=sys.stderr)
            else:
                print(f'Already downloaded {outfile}', file=sys.stderr)

            outdir = os.path.join(landir, f'OUT_{fname}')
            if not os.path.isdir(outdir):
                proc = subprocess.Popen(f'python -m wikiextractor.WikiExtractor {outfile} -o {outdir} -b 100M --no-templates'.split())
                output, error = proc.communicate()
                print(output, file=sys.stderr)
                print(error, file=sys.stderr)
        # cat OUT_*/*/wiki* > ../zhwiki.txt
        # cat enwiki.txt  | grep -v '</doc>' | grep -v '<doc id=' | shuf > enwikishuf.txt
        # lantxt = os.path.join(self.cachedir, f'{lan}.txt')
        # if not os.path.isfile(lantxt):
        #     print(f'concatenating to {lantxt}')
        #     with open(lantxt, 'w') as f:
        #         proc = subprocess.Popen(f'cat {landir}/OUT_*/*/wiki*'.split(), stdout=f)
        #         output, error = proc.communicate()
        #         # print(output, file=sys.stderr)
        #         print(error, file=sys.stderr)
    
    def headmb(self, lan, sizemb):
        assert lan in self.lans
        size = int(sizemb * 1000000)
        lantxt = os.path.join(self.cachedir, f'{lan}.txt')
        f = open(lantxt, 'rt', encoding="utf-8")
        sizedtxt = f.read(size)
        return sizedtxt

class BUCC20Corpus(object):
    dataurls = {
        'en-wiki': 'http://corpus.leeds.ac.uk/serge/bucc/en.ol.xz',
        'es-wiki': 'http://corpus.leeds.ac.uk/serge/bucc/es.ol.xz',
        'zh-wiki': 'http://corpus.leeds.ac.uk/serge/bucc/zh.ol.xz',
        'en-wac': 'http://corpus.leeds.ac.uk/serge/bucc/ukwac.ol.xz',
        'de-wac': 'http://corpus.leeds.ac.uk/serge/bucc/dewac.ol.xz',
        'fr-wac': 'http://corpus.leeds.ac.uk/serge/bucc/frwac.ol.xz',
        'ru-wac': 'http://corpus.leeds.ac.uk/serge/bucc/ruwac.ol.xz',
    }
    cachedir = os.path.expanduser('~/.cache/bucc20/corpus/')
    sizeddir = os.path.expanduser('~/.cache/bucc20/corpus/sized/')

    def __init__(self):
        sizeddir = BUCC20Corpus.sizeddir
        cachedir = BUCC20Corpus.cachedir
        if not os.path.isdir(cachedir):
            print(f'Making dir {cachedir}', file=sys.stderr)
            os.makedirs(cachedir, exist_ok=True)
        if not os.path.isdir(sizeddir):
            print(f'Making dir {sizeddir}', file=sys.stderr)
            os.makedirs(sizeddir, exist_ok=True)
    
    def check_and_dl_all(self):
        for lan in BUCC20Corpus.dataurls:
            self.check_and_dl(lan)

    def check_and_dl(self, lan):
        cachedir = BUCC20Corpus.cachedir
        url = BUCC20Corpus.dataurls[lan]
        outfile = os.path.join(cachedir, lan + '.xz')
        # print(f'Making cache dir {self.cachedir}', file=sys.stderr)
        if not os.path.isdir(cachedir):
            print(f'Making cache dir {cachedir}', file=sys.stderr)
            os.makedirs(cachedir, exist_ok=True)
        if not os.path.isfile(outfile):
            print(f'Downloading {outfile}', file=sys.stderr)
            proc = subprocess.Popen(['wget', url, '-O', outfile])
            output, error = proc.communicate()
            print(output, file=sys.stderr)
            print(error, file=sys.stderr)

    def headmb(self, lan, sizemb):
        size = int(sizemb * 1000000)
        xzfile = os.path.join(BUCC20Corpus.cachedir, lan + '.xz')
        if not os.path.isfile(xzfile):
            self.check_and_dl(lan)

        f = lzma.open(xzfile, 'rt', encoding="utf-8")
        sizedtxt = f.read(size)
        return sizedtxt

class EvalData(object):
    def eval_path(self, lanpair):
        pass

class MUSEEval(EvalData):
    _base = 'https://dl.fbaipublicfiles.com/arrival/dictionaries/'
    pairs = ['en-de', 'en-es', 'en-fr', 'en-ru', 'en-zh', 'en-fi', 'en-hu']
    cachedir = os.path.expanduser(os.path.join(cachebase, 'muse_dicts/'))
    types = {
        'full': '.txt',
        'train': '.0-5000.txt',
        'test': '.5000-6500.txt'      
    }

    def __init__(self, ):
        if not os.path.isdir(self.cachedir):
            print(f'Making cache dir {self.cachedir}', file=sys.stderr)
            os.makedirs(self.cachedir, exist_ok=True)
        
        for p in self.pairs:
            self.download(p)
    
    def download(self, p):
        for t in self.types:
            suff = self.types[t] 
            url = self._base + f'{p}{suff}'
            outfile = os.path.join(self.cachedir, f'{p}{suff}')
            if not os.path.isfile(outfile):
                print(f'Downloading {url}', file=sys.stderr)
                proc = subprocess.Popen(['wget', url, '-O', outfile])
                output, error = proc.communicate()
    
    def eval_path(self, lanpair, type='full'):
        if lanpair not in self.pairs:
            print(f'lanpair {lanpair} not in {self.pairs}, try downloading')
            self.download(lanpair)
        return os.path.join(self.cachedir, lanpair + self.types[type])

class BUCC20Eval(EvalData):
    dataurls = {
        'de-en': 'https://comparable.limsi.fr/bucc2020/tr/de-en-1-5000-training.txt',
        'es-en': 'https://comparable.limsi.fr/bucc2020/tr/es-en-1-5000-training.txt',
        'fr-en': 'https://comparable.limsi.fr/bucc2020/tr/fr-en-1-5000-training.txt',
        'ru-en': 'https://comparable.limsi.fr/bucc2020/tr/ru-en-1-5000-training.txt',
        'zh-en': 'https://comparable.limsi.fr/bucc2020/tr/zh-en-1-4500-training.txt',

        'en-de': 'https://comparable.limsi.fr/bucc2020/tr/en-de-1-5000-training.txt',
        'en-es': 'https://comparable.limsi.fr/bucc2020/tr/en-es-1-5000-training.txt',
        'en-fr': 'https://comparable.limsi.fr/bucc2020/tr/en-fr-1-5000-training.txt',
        'en-ru': 'https://comparable.limsi.fr/bucc2020/tr/en-ru-1-5000-training.txt',
        'en-zh': 'https://comparable.limsi.fr/bucc2020/tr/en-zh-1-5000-training.txt',
    }

    cachedir = os.path.expanduser('~/.cache/bucc20/train/')
    
    def __init__(self, ):
        cachedir = BUCC20Eval.cachedir
        if not os.path.isdir(cachedir):
            print(f'Making cache dir {cachedir}', file=sys.stderr)
            os.makedirs(cachedir, exist_ok=True)
        
        for lanpair in BUCC20Eval.dataurls:
            url = BUCC20Eval.dataurls[lanpair]
            outfile = os.path.join(cachedir, lanpair + '.txt')
            if not os.path.isfile(outfile):
                print(f'Downloading {url}', file=sys.stderr)
                proc = subprocess.Popen(['wget', url, '-O', outfile])
                output, error = proc.communicate()
    
    def eval_path(self, lanpair):
        return os.path.join(BUCC20Eval.cachedir, lanpair + '.txt')


def train_bpe_tokenizer(train_data_paths, model_path, vocab_size, limit_alphabet=100, min_frequency=10):
    trainer = BpeTrainer(special_tokens=["[UNK]"], min_frequency=min_frequency,
                         vocab_size=vocab_size, limit_alphabet=limit_alphabet)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]", fuse_unk=True))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Metaspace()
    print(f'data: {train_data_paths}', file=sys.stderr)
    tokenizer.train(train_data_paths, trainer)
    tokenizer.save(model_path)
    return tokenizer

def train_word_tokenizer(train_data_paths, model_path, vocab_size, limit_alphabet=100, min_frequency=10):
    trainer = WordLevelTrainer(special_tokens=["[UNK]"], min_frequency=min_frequency,
                               vocab_size=vocab_size, limit_alphabet=limit_alphabet)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    # tokenizer.pre_tokenizer = Metaspace()
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Lowercase()
    print(f'data: {train_data_paths}', file=sys.stderr)
    tokenizer.train(train_data_paths, trainer)
    tokenizer.save(model_path)
    return tokenizer

def tokenize(data_path: str, tokenizer: Tokenizer):
    t1 = time.perf_counter()
    with open(data_path, 'r') as f:
        all = f.readlines()
    t2 = time.perf_counter()
    print(f'reading {data_path} took: {t2 - t1 :.3f}', file=sys.stderr)
    alljoined = "".join(all)

    print(f'number of chars: {len(alljoined)}', file=sys.stderr)
    all = alljoined.split('\n')
    print(f'number of lines: {len(all)}', file=sys.stderr)

    t1 = time.perf_counter()
    tokened = tokenizer.encode_batch(all)
    t2 = time.perf_counter()
    print(f'encode_batch took: {t2 - t1 :.3f}', file=sys.stderr)
    return tokened

def text_to_cooc(tokened, tokenizer: Tokenizer, width=6):
    ids = [t.ids for t in tokened]
    joined_ids = list(itertools.chain(*ids))
    print('num words: ', len(joined_ids))

    t1 = time.perf_counter()
    cooc = cooc_count.cooc(np.array(joined_ids, dtype=np.int64), width=width, vocab_size=tokenizer.get_vocab_size())
    t2 = time.perf_counter()
    print(f'constructing cooc took: {t2 - t1 :.3f}', file=sys.stderr)
    return cooc

class Corpus(object):
    def __init__(self, datapath, basepath,
                 tokentype, vocab_size, limit_alphabet, min_frequency,
                 vectorize: str, width: int, dim: int, adaptive=False, write_vecs=False):
        self.adaptive = adaptive
        self.write_vecs = write_vecs
        self.dim = int(dim)
        self.width = int(width)
        self.vectorize = vectorize
        self.datapath = datapath
        os.makedirs(basepath, exist_ok=True)
        self.model_path = os.path.join(basepath, 'model.json')
        if tokentype == 'WordLevel':
            self.tokenizer = train_word_tokenizer([self.datapath], self.model_path, vocab_size, limit_alphabet, min_frequency)
        elif tokentype == 'BPE':
            self.tokenizer = train_bpe_tokenizer([self.datapath], self.model_path, vocab_size, limit_alphabet, min_frequency)
        else:
            raise Exception(f'{tokentype} not recognized')

        self.tokened = tokenize(self.datapath, self.tokenizer)
        self.tokened_out = os.path.join(basepath, 'c.tok')
        self.ids_out = os.path.join(basepath, 'c.ids')
        t1 = time.perf_counter()
        if vectorize == 'fasttext' or vectorize == 'word2vec':
            with open(self.tokened_out, 'w') as f:
                # basic tokenizer does not output UNK for unknown words, leading to all words being used
                f.writelines([' '.join([self.tokenizer.id_to_token(id) for id in li.ids]) for li in self.tokened])
            # with open(self.ids_out, 'w') as f:
            #     f.writelines([' '.join([str(id) for id in li.ids]) for li in self.tokened])
        t2 = time.perf_counter()
        print(f'writing tokened took: {t2 - t1 :.3f}', file=sys.stderr)

        self.vecpath = os.path.join(basepath, 'c.vec')
        self.vecs = {}
        self.Co = text_to_cooc(self.tokened, self.tokenizer, width=self.width)
        if vectorize == 'fasttext':
            self._fasttext_vecs()
        elif vectorize == 'word2vec':
            self._word2vec_vecs()
        elif vectorize == 'sim_svd':
            self._sim_vecs()
        elif vectorize == 'trunc':
            self._count_vecs()
        else:
            raise Exception('vectorize type not recognized')

    def _write_vectors(self, m):
        self.vec = m
        with open(self.vecpath, 'w') as f:
            vs = self.tokenizer.get_vocab_size()
            print('%d %d' % m.shape, file=f)
            for i in range(vs):
                print(self.tokenizer.id_to_token(i) + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=f)

    def _sim_vecs(self, alpha=0.5, beta=1):
        maxdim = min(self.Co.shape[1], 10000)
        Cot = self.Co[:, :maxdim]
        u, s, _ = np.linalg.svd(np.power(Cot, alpha), full_matrices=False)
        u = u[:, :self.dim]*np.power(s[:self.dim], beta)
        self.vecs['sim_svd'] = u
        if self.write_vecs:
            self._write_vectors(mat)

    def _fasttext_vecs(self, epoch=5):
        import fasttext
        # https://fasttext.cc/docs/en/options.html
        # two common corrections for fasttext
        if self.adaptive:
            lradapt = 0.1 / np.power(self.dim / 50, 0.5)
            mbsize = os.stat(self.tokened_out).st_size / 1e6
            epoch = 5 if mbsize > 300 else int(5 * np.sqrt(300 / mbsize))
            config = dict(model='skipgram', lr=lradapt, dim=self.dim, ws=self.width, epoch=epoch)
        else:
            config = dict(model='skipgram', lr=0.05, dim=self.dim, ws=self.width, epoch=5)
        
        print(config)
        # config = dict(model='skipgram', lr=0.05, dim=self.dim, ws=self.width, epoch=epoch, minn=0, maxn=0)
        model = fasttext.train_unsupervised(self.tokened_out, thread=8, **config)
        
        mat = np.zeros((self.tokenizer.get_vocab_size(), model.dim), dtype=float)
        for w in self.tokenizer.get_vocab():
            v = model.get_word_vector(w)
            i = self.tokenizer.token_to_id(w)
            mat[i] = v
        self.vecs['fasttext'] = mat
        if self.write_vecs:
            self._write_vectors(mat)
    
    def _word2vec_vecs(self, epoch=5):
        import fasttext
        # https://fasttext.cc/docs/en/options.html
        # just fasttext without subwords
        config = dict(model='skipgram', lr=0.05, dim=self.dim, ws=self.width, epoch=epoch, minn=0, maxn=0)
        model = fasttext.train_unsupervised(self.tokened_out, thread=8, **config)
        mat = np.zeros((self.tokenizer.get_vocab_size(), model.dim), dtype=float)
        for w in self.tokenizer.get_vocab():
            v = model.get_word_vector(w)
            i = self.tokenizer.token_to_id(w)
            mat[i] = v
        self.vecs['word2vec'] = mat
    
    def _count_vecs(self, alpha=0.5):
        mat = np.power(np.array(self.Co[:, :self.dim], dtype=float), alpha)
        self.vecs['trunc'] = mat
        if self.write_vecs:
            self._write_vectors(mat)

# eval = BUCC20Eval()
# bucc20.get_sized('zh-wiki', 2)
# bucc20.get_sized('zh-wiki', 4)
