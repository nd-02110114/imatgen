import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path, makedirs, getcwd
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure


from utils.preprocess import basis_translate, ase_atoms_to_image


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create basis images')

    parser.add_argument('--structure-path', default='../../dataset/raw/data_2020_03_03.h5',
                        type=str, help='path to cif data (relative path)')
    parser.add_argument('--csv-path', default='../../dataset/raw/data_2020_03_03.csv',
                        type=str, help='path to csv data (relative path)')
    parser.add_argument('--mp-ids', default='../../dataset/preprocess/mp_dataset_30000_2020_03/mp_ids.npy',
                        type=str, help='path to mp ids data (relative path)')
    parser.add_argument('--out-dir', '-o', default='../../dataset/preprocess/mp_dataset_30000_2020_03',
                        type=str, help='path for output directory')

    parser.add_argument('--fmt', default='cif',
                        type=str, help='format for structure data')
    parser.add_argument('--atom-list', default=None,
                        type=str, help='atom list')
    parser.add_argument('--jobs', default=-1,
                        type=str, help='cpu core for task')
    return parser.parse_args()


def main():
    # get args
    args = parse_arguments()

    # make output directory
    out_dir = args.out_dir
    out_dir_path = path.normpath(path.join(getcwd(), out_dir))
    makedirs(out_dir_path, exist_ok=True)
    cell_dir_path = path.join(out_dir_path, 'basis_image')
    makedirs(cell_dir_path, exist_ok=True)

    # load raw dataset
    structure_path = path.normpath(path.join(getcwd(), args.structure_path))
    csv_path = path.normpath(path.join(getcwd(), args.csv_path))
    mp_ids_path = path.normpath(path.join(getcwd(), args.mp_ids))
    structure_data = h5py.File(structure_path, "r")
    table_data = pd.read_csv(csv_path, index_col=False)
    mp_ids = np.load(mp_ids_path)
    table_data = table_data[table_data['material_id'].isin(mp_ids)]
    assert len(mp_ids) == len(table_data)

    basis_nbins = 64
    # 余分に領域を確保しておく
    length = len(table_data['material_id'].values) * 5
    basis_csv = pd.DataFrame([], columns=['mp_id', 'element', 'image_name'],
                             index=range(length))
    df_index = 0
    for i, mp_id in tqdm(enumerate(table_data['material_id'].values)):
        crystal = Structure.from_str(structure_data[mp_id].value, args.fmt)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        basis_atoms = basis_translate(ase_atoms)
        basis_image, channellist = ase_atoms_to_image(basis_atoms, basis_nbins, args.atom_list, args.jobs)

        # save basis image
        mp_id_for_path = mp_id.replace('-', '_')
        for j, atom in enumerate(channellist):
            image_name = '{}_{}'.format(mp_id_for_path, atom)
            basis_save_path = path.join(cell_dir_path, image_name)
            image = np.array([basis_image[:, :, :, j]])
            np.save('{}.npy'.format(basis_save_path), image)
            basis_csv.iloc[df_index] = [mp_id, atom, image_name]
            df_index += 1

    # save
    basis_csv = basis_csv.dropna(how='any')
    basis_csv_save_path = path.join(out_dir_path, 'basis_image.csv')
    basis_csv.to_csv(basis_csv_save_path)

    return True


if __name__ == '__main__':
    main()
