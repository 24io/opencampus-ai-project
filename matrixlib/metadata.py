import numpy as np
from typing import TypeVar, Generic, Union


T = TypeVar('T')


class MatrixMetadata:
    size: int
    matrix_dimensions: int
    background_noise_range: 'DataRange[float]'
    noise_block_range: 'DataRange[float]'
    true_block_range: 'DataRange[float]'

    noise_background_density: np.ndarray
    noise_block_density: np.ndarray
    true_block_density: np.ndarray
    block_starts: np.ndarray

    def __init__(self, size: int, matrix_dimensions: int):
        self.size = size
        self.matrix_dimensions = matrix_dimensions

        self.noise_background_density = np.zeros(size)
        self.noise_block_density = np.zeros(size)
        self.true_block_density = np.zeros(size)
        self.block_starts = np.zeros((size, matrix_dimensions))

    def get_shape(self):
        return self.size, self.matrix_dimensions


class DataRange(Generic[T]):
    min: T
    max: T

    def __init__(self, min_val: T, max_val: T):
        self.min = min_val
        self.max = max_val
