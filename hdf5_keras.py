# coding=utf-8
import h5py
import numpy as np

with h5py.File("mytestfile.hdf5", "w") as f:
    dset = f.create_dataset("mydataset", (100,), dtype='i')
    grp = f.create_group("subgroup")
    dset2 = grp.create_dataset("another_dataset", (50,), dtype='f')
    dset3 = f.create_dataset('subgroup2/dataset_three', (10,), dtype='i')
    dataset_three = f['subgroup2/dataset_three']
    for name in f:
        print(name)
    