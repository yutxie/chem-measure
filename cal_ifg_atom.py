#
#  Original authors: Richard Hall and Guillaume Godin
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.

#
#
# Richard hall 2017
# IFG main code
# Guillaume Godin 2017
# refine output function
# astex_ifg: identify functional groups a la Ertl, J. Cheminform (2017) 9:36
from rdkit import Chem
from collections import namedtuple
import rdkit
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
import pandas as pd
import pathlib
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', '-m',
        type=str, default='aae',
        help='model name : aae vae latentgan reinvent organ char_rnn'
    )
    parser.add_argument(
        '--epoch', '-e',
        type=int, default=100,
        help='epoch that SMILES was sampled'
    )
    parser.add_argument(
        '--n_jobs', '-n', type=int, default=40,
        help='number of processes to use'
    )

    parser.add_argument(
        '--input_file','-i', type=str,
        default='./',
        help='path to sampled files'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default='./SampledDatasetCano.csv',
        help='path to save files'
    )
    parser.add_argument(
        '--sample_id', '-si', type=int,
        default='0',
        help='Index of sampling'
    )
    return parser


def merge(mol, marked, aset):
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)

# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)

def identify_functional_groups(smi):
    try:
#         mol = Chem.MolFromSmiles(smi)
        mol = smi
        marked = set()
        #mark all heteroatoms in a molecule, including halogens
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in (6,1): # would we ever have hydrogen?
                marked.add(atom.GetIdx())

        #mark the four specific types of carbon atom
        for patt in PATT_TUPLE:
            for path in mol.GetSubstructMatches(patt):
                for atomindex in path:
                    marked.add(atomindex)

        #merge all connected marked atoms to a single FG
        groups = []
        while marked:
            grp = set([marked.pop()])
            merge(mol, marked, grp)
            groups.append(grp)

        #extract also connected unmarked carbon atoms
        ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
        ifgs = []
        for g in groups:
            uca = set()
            for atomidx in g:
                for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                    if n.GetAtomicNum() == 6:
                        uca.add(n.GetIdx())
            #ifgs.append(ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True), type=Chem.MolFragmentToSmiles(mol, g.union(uca),canonical=True)))
            ifgs.append(Chem.MolFragmentToSmiles(mol, g, canonical=True))
        return ifgs
    except:
        return None

def CollectFG(smis, n_jobs=1):
    ifgs = []
    with Pool(n_jobs) as pool:
        identify_functional_groups_p = partial(identify_functional_groups)
        ifgs = [x for x in pool.imap_unordered(identify_functional_groups_p, smis) if x is not None]
    ifg_1col = []
    for i in ifgs:
        for j in i:
            ifg_1col.append(j)

    return ifg_1col
def main(config):
    #model = config.model
    # epoch = config.epoch
    # sample_id = config.sample_id
    n_jobs = config.n_jobs
    #sample_id = config.sample_id
    # output = config.output
    # file = '%s_%d_combine.csv' % (model, sample_id)
    # file = 'gdb13_purged.csv'
    file = config.input_file+'.csv'
    mols = pd.read_csv(file, names=['SMILES','model'])
    print('%s was loaded!\n'%(file))
    smis =  list(mols['SMILES'])
    print('smis was retrieved')
    ifgs = CollectFG(smis, n_jobs)
    ifgs = pd.DataFrame(ifgs, columns=['ifgs'])
    ifgs.drop_duplicates('ifgs', inplace=True)
    # pd_ifg.to_csv('%s_%d_ifg.csv' % (model, sample_id), index=None)
    ifgs.to_csv(config.input_file+'_ifgs_atom.csv', index=None)



if __name__ == "__main__":
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    main(config)