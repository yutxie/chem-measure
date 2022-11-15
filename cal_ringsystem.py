import rdkit
from rdkit import Chem
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


def file_collection(model, epoch):
    file_set = []
    for i in range(8):
        file_set.append('%s_model_s_%d_%d.csv.cano' % (model, epoch, i))
    return file_set


def open_file(file_set, path):
    dataset = []
    file_path = pathlib.Path(path, file_set)
    with open(file_path, 'r') as f:         # gzip.open(path) as smi:
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.split()
            dataset.append(line[0])
    return dataset


def smile_canonical(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        smi_cano = Chem.MolToSmiles(mol, isomericSmiles=False)
        return smi_cano
    except:
        print(smi + "was not valid SMILES\n")
        return None


def set_canonical(n_jobs, output, file_set):
    global dataset
    with Pool(n_jobs) as pool:
        smile_canonical_p = partial(smile_canonical)
        dataset_cano = [x for x in pool.imap_unordered(smile_canonical_p, dataset) if x is not None]
    dataset = []
    dataset_cano = pd.DataFrame(dataset_cano, columns=['SMILES'])
    dataset_cano = dataset_cano.drop_duplicates('SMILES')
    dataset_cano.to_csv(file_set+'.cano', index=None)

def combine(output, file_set):
    dataset_list = []
    for file in file_set:    
        dataset_list.append(pd.read_csv(file, names=['SMILES']))
    pd_combine =  pd.concat(dataset_list)
    pd_combine = pd_combine.drop_duplicates('SMILES')
    pd_combine.to_csv(output, index=None)
# def AtomAppendRing(mol, systems):
#     atoms = mol.GetAtoms()
#     natoms = len(atoms)
#     for atom_i in range(natoms):
#         for ring in systems:
#             if atom_i not in ring:
#                 continue


def GetRingSystems(smi, includeSpiro=False):
    try:
#         mol = Chem.MolFromSmiles(smi)
        mol = smi
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ringAts = set(ring)
            nSystems = []
            for system in systems:
                nInCommon = len(ringAts.intersection(system))
                if nInCommon and (includeSpiro or nInCommon > 1):
                    ringAts = ringAts.union(system)
                else:
                    nSystems.append(system)
            nSystems.append(ringAts)
            systems = nSystems
        ring_smi = []
        for system in systems:
            frag_smi = Chem.MolFragmentToSmiles(mol, system, canonical=True)
            ring_smi.append(frag_smi)
        return ring_smi
    except:
        return None

def CollectRingSystems(smis, n_jobs=1):
    ringsystem = []
    with Pool(n_jobs) as pool:
        GetRingSystems_p = partial(GetRingSystems)
        rings = [x for x in pool.imap_unordered(GetRingSystems_p, smis) if x is not None]
    for ring_list in rings:
        for ring in ring_list:
            ringsystem.append(ring)
    #     rings = GetRingSystems(smis[i])
    #     ringsystem += rings
    #     if i % 1000 == 0:
    #         print('%d SMILES have been processed!'%(i))
    #         ringsystem = list(set(ringsystem))
    return ringsystem

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
    ringsystem =  CollectRingSystems(smis, n_jobs)
    ringsystem = pd.DataFrame(ringsystem, columns=['rings'])
    ringsystem.drop_duplicates('rings', inplace=True)
    # pd_ringsystem.to_csv('%s_%d_ringsystem.csv' % (model, sample_id), index=None)
    ringsystem.to_csv(config.input_file+'_ringsystem.csv', index=None)




if __name__ == '__main__':
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    dataset = []
    main(config)