# Scripts for preparing MP subset data

## Usage
Please set the YYYY_MM_DD which the data was created on.

### 1. Load raw mp data

```
$ python get_raw_mp_data.py
```

### 2. Get valid mp ids

```
$ PYTHONPATH=../.. python get_mp_ids.py --structure-path ../../dataset/raw/data_YYYY_MM_DD.h5 \
                                        --csv-path ../../dataset/raw/data_YYYY_MM_DD.csv \
                                        -o ../../dataset/preprocess/mp_dataset_30000_YYYY_MM_DD
```

### 3. Create Cell and Basis images

**Caution** : Basis images requires a large file volume

```
$ PYTHONPATH=../.. python get_cell_image.py --structure-path ../../dataset/raw/data_YYYY_MM_DD.h5 \
                                            --csv-path ../../dataset/raw/data_YYYY_MM_DD.csv \
                                            --mp-ids ../../dataset/preprocess/mp_dataset_30000_YYYY_MM_DD/mp_ids.npy
                                            -o ../../dataset/preprocess/mp_dataset_30000_YYYY_MM_DD
$ PYTHONPATH=../.. python get_basis_image.py --structure-path ../../dataset/raw/data_YYYY_MM_DD.h5 \
                                            --csv-path ../../dataset/raw/data_YYYY_MM_DD.csv \
                                            --mp-ids ../../dataset/preprocess/mp_dataset_30000_YYYY_MM_DD/mp_ids.npy
                                            -o ../../dataset/preprocess/mp_dataset_30000_YYYY_MM_DD
```