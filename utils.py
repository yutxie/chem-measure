import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.metrics import pairwise_distances

def fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

def fingerprints(mols):
    return [fingerprint(mol) for mol in mols]

def similarities_tanimoto(fp, fps):
    return DataStructs.BulkTanimotoSimilarity(fp, fps)

def similarity_matrix_tanimoto(fps1, fps2):
    similarities = [DataStructs.BulkTanimotoSimilarity(fp, fps2) for fp in fps1]
    return np.array(similarities)

def latent_vectors(mols, dataset):
    idxs = [int(mol.GetProp('Index')) for mol in mols]
    vecs = [dataset.vecs[i] for i in idxs]
    return vecs

def similarity_matrix_latent(vecs1, vecs2):
    vecs1 = np.array(vecs1)
    vecs2 = np.array(vecs2)
    dist = pairwise_distances(vecs1, vecs2, metric='euclidean')
    sims = 1. / (1. + dist)
    return sims