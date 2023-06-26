# Copyright (c) Meta Platforms, Inc. and affiliates.
import subprocess
from subprocess import check_output
import os
import embeddings

class VecMap:
    """
    wrapper for vecmap https://github.com/artetxem/vecmap
    assumes vecmap is in the directory ./vecmap
    """
    def __init__(self, srcvec, tgtvec, dictpath, outdir, config):
        self.srcvec = srcvec
        self.tgtvec = tgtvec
        self.outdir = outdir
        self.dictpath = dictpath
    
        self.flags = ''
        self.config = config

        def add_flag(v):
            self.flags += ' ' + v

        add_flag('--verbose')
        add_flag('--orthogonal')
        # default is 50, but I want faster
        add_flag('--stochastic_interval 3')
        add_flag('--csls 10')
        if dictpath is not None:
            add_flag(f'--validation {dictpath}')

        logdir = os.path.join(self.outdir, 'vecmap.log')
        add_flag(f'--log {logdir}')

        if config.supervision == 'identical':
            add_flag('--identical')
        elif config.supervision == 'init_identical':
            add_flag('--init_identical')
        elif config.supervision == 'numeral':
            add_flag('--init_numeral')
        elif config.supervision == 'unsupervised':
            add_flag('--init_unsupervised')
            add_flag('--unsupervised')
        else:
            raise Exception('invalid type of supervision: ' + config.supervision)
        
        # if config.init_dict:
        #     add_flag(f'--dictionary {config.init_dict}')

    def run(self):
        srcvec = self.srcvec
        tgtvec = self.tgtvec
        srcvec_out = os.path.join(self.outdir, 'src.out.vec')
        tgtvec_out = os.path.join(self.outdir, 'tgt.out.vec')

        cmd = f'python vecmap/map_embeddings.py {srcvec} {tgtvec} {srcvec_out} {tgtvec_out} {self.flags} --verbose'
        print(cmd)
        process = subprocess.Popen(cmd.split())
        output, error = process.communicate()

    def vecs(self):
        srcvec_out = os.path.join(self.outdir, 'src.out.vec')
        tgtvec_out = os.path.join(self.outdir, 'tgt.out.vec')
        srcfile = open(srcvec_out, encoding='utf-8', errors='surrogateescape')
        tgtfile = open(tgtvec_out, encoding='utf-8', errors='surrogateescape')
        
        src_words, srcvec = embeddings.read(srcfile)
        tgt_words, tgtvec = embeddings.read(tgtfile)
        return srcvec, tgtvec
    
    def eval(self, dictpath):
        srcvec_out = os.path.join(self.outdir, 'src.out.vec')
        tgtvec_out = os.path.join(self.outdir, 'tgt.out.vec')
        cmd = f'python vecmap/eval_translation.py {srcvec_out} {tgtvec_out} -d {dictpath} --retrieval csls -k 10'
        print(cmd)
        out = check_output(cmd.split())

        def cov_acc(sout):
            toks = sout.decode('utf8').replace(': ', ' ').replace(':', ' ').split()
            print(sout)
            cov = float(toks[1].strip('%'))
            acc = float(toks[3].strip('%'))
            return ({'accuracy': acc, 'coverage': cov})
    
        return cov_acc(out)

