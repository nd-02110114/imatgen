# Scripts for training materials generator

## Usage

Please set the `YYYY_MM_DD` which the data was created on.

```
$ PYTOHNPATH=.. python train.py \
                --data-path ../dataset/preprocess/mp_dataset_30000_YYYY_MM_DD \
                --raw_data_dir ../dataset/raw/data_YYYY_MM_DD.csv \
                -o result \
                --gpu
```
