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


def add_noise(
        matrix_array: np.ndarray,
        density_array: np.ndarray,
        density_range: (float, float),
        value_range: (float, float),
) -> None:
    # unpack and validate number of matrices and dimension
    number_of_matrices, dimension, dimension_check = matrix_array.shape
    if dimension_check != dimension:
        raise ValueError("Dimension mismatch")

    # unpack ranges
    density_low: float = density_range[0]
    density_high: float = density_range[1]
    value_low: float = value_range[0]
    value_high: float = value_range[1]

    # create some random noise values (some might be overridden later)
    for n in range(number_of_matrices):
        noise_density: float = np.random.uniform(density_low, density_high)
        density_array[n] = noise_density
        for j in range(dimension):
            for i in range(j):
                if np.random.random() < noise_density:
                    value: float = np.random.uniform(value_low, value_high)
                    matrix_array[n][j][i] = value
                    matrix_array[n][i][j] = value
