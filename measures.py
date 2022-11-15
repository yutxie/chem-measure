import rdkit
import random
import numpy as np
import more_itertools as mit
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

CHUNK_SIZE = 1000


class Measure():
    def __init__(self):
        pass

    def update(self, mols=[]):
        raise NotImplementedError
        
    def measure(self, mols=[]):
        self.update(mols)
        return self.report()

    def report(self):
        raise NotImplementedError
    
    
class GoldenStandard(Measure):
    def __init__(self):
        super().__init__()
        self.target_set = set()
        
    def update(self, mols=[]):
        for mol in mols:
            self.target_set.add(mol.GetProp('Target'))
        
    def report(self):
        return len(self.target_set)


class DissimilarityBasedMeasure(Measure):
    def __init__(self, 
        vectorizer=None, 
        sim_mat_func=None, 
        *args, **kargs
    ):
        super().__init__(*args, **kargs)
        self.vectorizer = vectorizer
        self.sim_mat_func = sim_mat_func
        self.dis_mat = np.zeros((0, 0))
        self.vecs = []

    def update(self, mols=[], is_vec=False):
        n = len(self.vecs)
        new_n = n + len(mols)
        new_dis_mat = np.zeros((new_n, new_n))
        new_dis_mat[:n, :n] = self.dis_mat
        
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        sim_mat = self.sim_mat_func(vecs, vecs)
        new_dis_mat[n:, n:] = 1 - np.array(sim_mat)
        
        if n > 0:
            sim_mat = self.sim_mat_func(self.vecs, vecs)
            new_dis_mat[:n, n:] = 1 - np.array(sim_mat)
            new_dis_mat[n:, :n] = 1 - np.array(sim_mat).T
        
        self.dis_mat = new_dis_mat
        self.vecs += vecs

    def report(self):
        raise NotImplementedError
        
def diversity_map(args):
    vec, vecs, sim_mat_func = args
    dists = sim_mat_func([vec], vecs)[0]
    return dists.mean()

class Diversity(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        
    def measure(self, mols=[], is_vec=False):
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        args = zip(vecs, [vecs]*len(vecs), [self.sim_mat_func]*len(vecs))
        avg_dists = process_map(diversity_map, args, chunksize=CHUNK_SIZE)
        return 1. - np.sum(avg_dists) / len(avg_dists)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return (m.sum(axis=-1) / (n - 1)).mean().item()

class SumDiversity(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return (m.sum(axis=-1) / (n - 1)).sum().item()
    
class Bottleneck(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        m = m + np.eye(n) * 1e9
        return m.min().item()
    
def sumbot_map(args):
    vec, vecs, sim_mat_func = args
    dists = 1.-sim_mat_func([vec], vecs)[0]
    return np.partition(dists, 1)[1]

class SumBottleneck(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        
    def measure(self, mols=[], is_vec=False):
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        
        args = zip(vecs, [vecs]*len(vecs), [self.sim_mat_func]*len(vecs))
        min_dists = process_map(sumbot_map, args, chunksize=CHUNK_SIZE)
        return np.sum(min_dists)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        m = m + np.eye(n) * 1e9
        return m.min(axis=-1).sum().item()

class Diameter(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return m.max().item()

class SumDiameter(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return m.max(axis=-1).sum().item()
    
class DPP(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        
    def report(self):
        sim_mat = 1 - self.dis_mat
        return np.linalg.det(sim_mat)

    
class ReferenceBasedMeasure(Measure):
    def __init__(self):
        super().__init__()

import pandas as pd
from cal_ifg_atom import CollectFG
from cal_ringsystem import CollectRingSystems
class NFragment(ReferenceBasedMeasure):
    def __init__(self, frag=None):
        super().__init__()
        if  frag == 'FG': self.collect_fn = CollectFG
        elif frag == 'RS': self.collect_fn = CollectRingSystems
        self.df = pd.DataFrame(columns=['frags'])
        
    def update(self, mols=[]):
        frags = self.collect_fn(mols)
        self.df = self.df.append(pd.DataFrame(frags, columns=['frags']))
        
    def report(self):
        self.df.drop_duplicates('frags', inplace=True)
        return len(self.df)
    
def bm_map(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    smi = Chem.MolToSmiles(core)
    return smi

from rdkit.Chem.Scaffolds import MurckoScaffold
class NBM(ReferenceBasedMeasure):
    def __init__(self):
        super().__init__()
        self.smiles = set()
        
    def measure(self, mols=[]):
        smiles = process_map(bm_map, mols, chunksize=CHUNK_SIZE)
        return len(set(smiles))
        
    def update(self, mols=[]):
        for mol in tqdm(mols):
            core = MurckoScaffold.GetScaffoldForMol(mol)
            smi = Chem.MolToSmiles(core)
            self.smiles.add(smi)
    
    def report(self):
        return len(self.smiles)

def get_circles(args, silent=True):
    vecs, sim_mat_func, t = args
    
    circs = []
    for vec in tqdm(vecs, disable=silent):
        if len(circs) > 0:
            dists = 1. - sim_mat_func([vec], circs)
            if dists.min() <= t: continue
        circs.append(vec)
    return circs

class NCircles(Measure):
    def __init__(self, 
        vectorizer=None, 
        sim_mat_func=None, 
        threshold=0.75,
        *args, **kargs
    ):
        super().__init__(*args, **kargs)
        self.vectorizer = vectorizer
        self.sim_mat_func = sim_mat_func
        self.t = threshold
        self.vecs = []
        
    def measure(self, mols=[], is_vec=False, n_chunk=64):
        # print('entering the func')
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        # print('splitting the list')
        
        for i in range(3):
            # print(n_chunk//(2**i))
            vecs_list = [list(c) for c in mit.divide(n_chunk//(2**i), vecs)]
            # print('parallelly computing n circles')
            args = zip(vecs_list, 
                       [self.sim_mat_func]*len(vecs_list), 
                       [self.t]*len(vecs_list))
            # print(len(vecs_list[0]))
            circs_list = process_map(get_circles, args)
            # print(len(circs_list[0]))
            vecs = [c for ls in circs_list for c in ls]
            random.shuffle(vecs)
        vecs = get_circles(
            (vecs, self.sim_mat_func, self.t), 
            silent=False)
        return len(vecs), vecs

    def update(self, mols=[], is_vec=False):
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        for vec in tqdm(vecs):
            if len(self.vecs) > 0:
                dists = 1. - self.sim_mat_func([vec], self.vecs)
                if dists.min() <= self.t: continue
            self.vecs.append(vec)

    def report(self):
        return len(self.vecs)
    
class Richness(Measure):
    def __init__(self):
        super().__init__()
        self.smiles = set()
        
    def update(self, mols=[]):
        for mol in mols:
            smi = Chem.MolToSmiles(mol)
            self.smiles.add(smi)
    
    def report(self):
        return len(self.smiles)