import h5py

with h5py.File('/mnt/data/nyudepthv2/train/bedroom_0004/01081.h5', 'r') as f:
    # Print the shape of each dataset in the HDF5 file
    for key in f.keys():
        print(f"{key}: {f[key].shape}")