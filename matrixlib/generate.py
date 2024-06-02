import numpy as np


def __init_zero_matrices(number_of_matrices: int, dimension: int) -> np.array:
    total_number_of_entries = number_of_matrices * dimension * dimension
    print(f"Generating matrices with a total number of {total_number_of_entries} entries ({number_of_matrices} {dimension}x{dimension} matrices)")
    generated_matrix = np.zeros((number_of_matrices, dimension, dimension))
    size_of_single_entry = generated_matrix.itemsize
    mem_size = total_number_of_entries * size_of_single_entry
    print(f"Matrix for element size {size_of_single_entry} bytes is a total of {mem_size} bytes ({mem_size / (1024 * 1024)} MiB)")
    return generated_matrix


def __init_metadata(number_of_matrices: int, dimension: int) -> (np.ndarray, np.ndarray, np.ndarray):
    noise_densities: np.ndarray = np.zeros(number_of_matrices, dtype=float)
    block_densities: np.ndarray = np.zeros(number_of_matrices, dtype=float)
    block_starts: np.ndarray = np.zeros((number_of_matrices, dimension), dtype=bool)
    return noise_densities, block_densities, block_starts
