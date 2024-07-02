import h5py
import numpy as np


def save_to_hdf5(dataset, filename: str):
    """
    Save a dataset to a HDF5 file.
    :param dataset: tensorflow dataset
    :param filename: String ending with .h5
    :return: None
    """
    bands_list = []
    labels_list = []

    for band, label in dataset:
        bands_list.append(band.numpy())
        labels_list.append(band.numpy())

    with h5py.File(filename, 'w') as f:
        f.create_dataset('labels', data=np.array(labels_list))


def load_from_hdf5(filename: str):
    """
    Load a dataset from a HDF5 file.
    :param filename: String ending with .h5
    :return: Numpy arrays of bands and labels
    """
    with h5py.File(filename, 'r') as f:
        bands = np.array(f['bands'])
        labels = np.array(f['labels'])
    return bands, labels
