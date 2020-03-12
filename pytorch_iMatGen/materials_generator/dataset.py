import torch
import functools
import numpy as np
import pandas as pd
from os import path
from torch.utils.data import Dataset


class MaterialsGeneratorDataset(Dataset):
    """
    Wrapper for a dataset
    """

    def __init__(self, mp_ids, data_dir, raw_data_dir):
        """
        Args:
            mp_ids (List): materials ids for cell images
            data_dir (string): path for preprocessed data
            raw_data_dir (string): path for raw csv data
        """
        self.mp_ids = mp_ids
        self.data_dir = data_dir
        self.formation_energy_data = pd.read_csv(raw_data_dir)
        self.cell_image_df = pd.read_csv(path.join(data_dir, 'cell_image.csv'))
        self.basis_image_df = pd.read_csv(path.join(data_dir, 'basis_image.csv'))

    def __len__(self):
        return len(self.mp_ids)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        mp_id = self.mp_ids[idx]
        cell_image_name = self.cell_image_df[self.cell_image_df['mp_id'] == mp_id]['image_name'].values
        basis_image_name = self.basis_image_df[self.basis_image_df['mp_id'] == mp_id]['image_name'].values

        # load encoded cell image
        cell_vector = np.zeros((1, 200))
        cell_image_encoded = np.load(path.join(self.data_dir, 'cell_image_encode', '{}.npy'.format(cell_image_name[0])))
        cell_vector[:, 0:len(cell_image_encoded)] = cell_image_encoded

        # load encoded cell image
        basis_vector = np.zeros((5, 200))
        for i, name in enumerate(basis_image_name):
            basis_image_encoded = np.load(path.join(self.data_dir, 'basis_image_encode', '{}.npy'.format(name)))
            basis_vector[i] = basis_image_encoded

        vector = np.concatenate([basis_vector, cell_vector], axis=0)
        # add a new axis
        vector = vector.reshape((1, 6, 200))
        # reshape (channel, height, width) = (6, 1, 200)
        vector = np.transpose(vector, (1, 0, 2))

        # for formation energy task
        formation_energy = self.formation_energy_data[
            self.formation_energy_data['material_id'] == mp_id
        ]['formation_energy_per_atom'].values[0]
        label = 0 if formation_energy <= -0.5 else 1
        return torch.tensor(vector, dtype=torch.float), torch.tensor([label], dtype=torch.float)
