import rdkit
import numpy as np

from rdkit import Chem
from tqdm import tqdm


class Dataset():
    def __init__(self, file_name, size=10000):
        suppl = Chem.SDMolSupplier(file_name)
        self.mols = []
        for i, mol in enumerate(tqdm(suppl)):
            mol.SetProp('Index', str(i))
            self.mols.append(mol)
            if len(self.mols) == size: break
#         self.mols = [mol for mol in tqdm(suppl)]

        self.targets = np.unique([mol.GetProp('Target') for mol in self.mols]).tolist()
        self.targ2idxs = {targ : [i for i, mol in enumerate(self.mols) if mol.GetProp('Target') == targ] for targ in self.targets}
        
    def load_vecs(self, file_name):
        self.vecs = np.load(file_name)