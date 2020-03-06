import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path, makedirs, getcwd
from pymatgen.core.structure import Structure


def parse_arguments():
    parser = argparse.ArgumentParser(description='Getting basis images')
    # for data
    parser.add_argument('--structure-path', default='../dataset/raw/data_2020_03_03.h5',
                        type=str, help='path to cif data (relative path)')
    parser.add_argument('--csv-path', default='../dataset/raw/data_2020_03_03.csv',
                        type=str, help='path to csv data (relative path)')
    parser.add_argument('--out-dir', '-o', default='../dataset/preprocess/mp_dataset_2020_03',
                        type=str, help='path for output directory')
    parser.add_argument('--fmt', default='cif',
                        type=str, help='format for structure data')

    return parser.parse_args()


def main():
    # get args
    args = parse_arguments()

    # make output directory
    out_dir = args.out_dir
    out_dir_path = path.normpath(path.join(getcwd(), out_dir))
    makedirs(out_dir_path, exist_ok=True)

    # load raw dataset
    csv_path = path.normpath(path.join(getcwd(), args.csv_path))
    table_data = pd.read_csv(csv_path, index_col=False)
    structure_path = path.normpath(path.join(getcwd(), args.structure_path))
    structure_data = h5py.File(structure_path, "r")

    # loop
    mp_ids = []
    for mp_id in tqdm(table_data['material_id']):
        crystal = Structure.from_str(structure_data[mp_id].value, args.fmt)
        n_sites = len(crystal.sites)
        max_lattice_langth = max(crystal.lattice.lengths)
        # https://github.com/kaist-amsg/imatgen/issues/2
        if n_sites <= 20 and max_lattice_langth <= 10:
            mp_ids += [mp_id]

    # save
    save_path = path.join(out_dir, 'mp_ids.npy')
    np.save(save_path, mp_ids)

    return True


if __name__ == '__main__':
    main()
