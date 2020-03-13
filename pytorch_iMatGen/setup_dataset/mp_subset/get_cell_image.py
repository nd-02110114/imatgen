import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path, makedirs, getcwd
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


from utils.preprocess import cell_translate, ase_atoms_to_image


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create cell images')

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
    cell_dir_path = path.join(out_dir_path, 'cell_image')
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

    cell_nbins = 32
    length = len(table_data['material_id'].values)
    cell_csv = pd.DataFrame([], columns=['mp_id', 'crystal_system', 'space_group', 'image_name'],
                            index=range(length))
    for i, mp_id in tqdm(enumerate(table_data['material_id'].values)):
        crystal = Structure.from_str(structure_data[mp_id].value, args.fmt)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        cell_atoms = cell_translate(ase_atoms)
        cell_image, _ = ase_atoms_to_image(cell_atoms, cell_nbins, args.atom_list, args.jobs)

        # save cell image
        image_name = mp_id.replace('-', '_')
        cell_save_path = path.join(cell_dir_path, image_name)
        np.save('{}.npy'.format(cell_save_path), cell_image)

        # save basis image
        finder = SpacegroupAnalyzer(crystal)
        crystal_system = finder.get_crystal_system()
        space_group_num = finder.get_space_group_number()
        cell_csv.iloc[i] = [mp_id, crystal_system, space_group_num, image_name]

    # save
    cell_csv_save_path = path.join(out_dir_path, 'cell_image.csv')
    cell_csv.to_csv(cell_csv_save_path)

    return True


if __name__ == '__main__':
    main()
