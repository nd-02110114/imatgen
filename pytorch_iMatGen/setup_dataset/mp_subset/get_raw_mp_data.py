import argparse
from os import path, makedirs, getcwd
from datetime import datetime

import h5py
import pandas as pd
from pymatgen import MPRester


def parse_arguments():
    parser = argparse.ArgumentParser(description='Download mp dataset')

    parser.add_argument('--out', '-o', type=str, default='../../dataset/raw',
                        help='directory path to save the data')
    parser.add_argument('--name', '-n', type=str, default='data',
                        help='filename for csv and h5')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse the arguments.
    args = parse_arguments()

    # create save path
    target_dir = args.out
    out_dir_path = path.normpath(path.join(getcwd(), target_dir))
    makedirs(out_dir_path, exist_ok=True)
    filename = args.name
    time = datetime.now().strftime("%Y_%m_%d")
    out_csv_path = path.join(out_dir_path, filename + "_" + time + ".csv")
    out_h5_path = path.join(out_dir_path, filename + "_" + time + ".h5")

    # initialize MPRester
    MP_KEY = 'cRelC34nhXp1wf5H'
    mp_dr = MPRester(MP_KEY)

    # query
    criteria = {
        # https://discuss.materialsproject.org/t/how-to-exclude-poorly-converged-calculations/2523
        "warnings": {
            "$nin": [
                "Large change in a lattice parameter during relaxation.",
                "Large change in volume during relaxation."
            ]
        },
    }

    # property
    properties = [
        'material_id',
        'pretty_formula',
        'unit_cell_formula',
        'energy',
        'energy_per_atom',
        'formation_energy_per_atom',
        'band_gap',
        'efermi',
        'elasticity',
        'volume',
        'density',
        # https://wiki.materialsproject.org/Glossary_of_Terms
        # stable : e_above_hull < 0.000001(1e-6)
        'e_above_hull',
        'cif',
    ]

    # collect all data
    data = mp_dr.query(criteria=criteria, properties=properties)

    # save property data
    list_data = [list(dict_val.values())[:-1] for dict_val in data]
    df = pd.DataFrame(list_data, columns=properties[:-1])
    df.to_csv(out_csv_path)

    # save cif data
    with h5py.File(out_h5_path, 'w') as f:
        for value in data:
            f.create_dataset(value['material_id'], data=value['cif'])
