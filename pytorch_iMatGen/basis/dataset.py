import torch
import functools
import numpy as np
import pandas as pd
from os import path
from torch.utils.data import Dataset


class BasisImageDataset(Dataset):
    """
    Wrapper for a dataset
    """

    def __init__(self, mp_ids, data_dir, encode=False):
        """
        Args:
            mp_ids (List): materials ids for basis images
            data_dir (string): path for preprocessed data
            encode (Boolean): encoding or not
        """
        self.data_dir = data_dir
        self.encode = encode
        table = pd.read_csv(path.join(data_dir, 'basis_image.csv'))
        self.table = table[table['mp_id'].isin(mp_ids)]
        assert set(mp_ids).issubset(list(table['mp_id'].values))

    def __len__(self):
        return len(self.table)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        image_name = self.table.iloc[idx]['image_name']
        image = np.load(path.join(self.data_dir, 'basis_image', '{}.npy'.format(image_name)))
        if self.encode:
            return torch.tensor(image, dtype=torch.float), image_name

        return torch.tensor(image, dtype=torch.float)
