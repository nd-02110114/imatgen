import torch._tensor_docs
import functools
import numpy as np
from torch.utils.data import Dataset
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure


from utils.preprocess import cell_translate, ase_atoms_to_image


def cif_to_cell_image(position_info, nbins, all_atomlist, num_cores, fmt='cif'):
    crystal = Structure.from_str(position_info, fmt)
    ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
    translated_ase_atoms = cell_translate(ase_atoms)
    image, channellist = ase_atoms_to_image(translated_ase_atoms, nbins, all_atomlist, num_cores)
    return image


class CellImageDataset(Dataset):
    """
    Wrapper for a dataset
    """

    def __init__(self, cifs, table, transform=cif_to_cell_image,
                 nbins=32, allatom_list=None, num_cores=-1, fmt='cif',
                 random_seed=1234):
        """
        Args:
            # TODO: Consider about POSCAR case
            cifs (HDF5): HDF5 object of cif data.
            table (DataFrame): dataset with materials id and each property
            transform (callable, optional): Optional transform to be applied on a sample.
            nbins (int): number of bins in one dimension of image
            allatom_list (list): element lists for data.
            num_cores (int): number of process for loading data.
            fmt (str): type of the structure informatio.n (cif or POSCAR)
            random_seed (int): Random seed for shuffling the dataset.
        """
        self.mp_ids = table['material_id'].values
        self.cifs = cifs
        self.transform = transform
        self.nbins = nbins
        self.allatom_list = allatom_list
        self.num_cores = num_cores
        self.fmt = fmt

    def __len__(self):
        return len(self.mp_ids)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        mp_id = self.mp_ids[idx]
        image = self.transform(self.cifs[mp_id].value, self.nbins,
                               self.allatom_list, self.num_cores, self.fmt)
        image = np.transpose(image, (3, 0, 1, 2))
        return torch.tensor(image, dtype=torch.float)
