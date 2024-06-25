import h5py
import numpy as np


def save_to_hdf5(dataset, filename: str):
    """
    Save a dataset to a HDF5 file.
    :param dataset: tensorflow dataset
    :param filename: String ending with .h5
    :return: None
    """
    features_list = []
    labels_list = []

    for feature, label in dataset:
        features_list.append(feature.numpy())
        labels_list.append(feature.numpy())

    with h5py.File(filename, 'w') as f:
        f.create_dataset('features', data=np.array(features_list))
        f.create_dataset('labels', data=np.array(labels_list))


def load_from_hdf5(filename: str):
    """
    Load a dataset from a HDF5 file.
    :param filename: String ending with .h5
    :return: Numpy arrays of features and labels
    """
    with h5py.File(filename, 'r') as f:
        features = np.array(f['features'])
        labels = np.array(f['labels'])
    return features, labels
