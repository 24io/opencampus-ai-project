import numpy as np
import scipy.stats as stats
from typing import Tuple, Union, List

def __init_zero_matrices(number_of_matrices: int, dimension: int) -> np.array:
    total_number_of_entries = number_of_matrices * dimension * dimension
    print(
        f"Generating matrices with a total number of {total_number_of_entries} entries " +
        f"({number_of_matrices} {dimension}x{dimension} matrices)"
    )
    generated_matrix = np.zeros((number_of_matrices, dimension, dimension))
    size_of_single_entry = generated_matrix.itemsize
    mem_size = total_number_of_entries * size_of_single_entry
    print(
        f"Matrix for element size {size_of_single_entry} bytes is a total of " +
        f"{mem_size} bytes ({mem_size / (1024 * 1024)} MiB)"
    )
    return generated_matrix


def add_noise(
        matrix_array: np.ndarray,
        density_array: np.ndarray,
        density_min: float,
        density_max: float,
        value_min: float,
        value_max: float,
) -> None:
    # unpack and validate number of matrices and dimension
    number_of_matrices, dimension, dimension_check = matrix_array.shape
    if dimension_check != dimension:
        raise ValueError("Dimension mismatch")

    # create some random noise values (some might be overridden later)
    for n in range(number_of_matrices):
        noise_density: float = np.random.uniform(density_min, density_max)
        density_array[n] = noise_density
        for j in range(dimension):
            for i in range(j):
                if np.random.random() < noise_density:
                    value: float = np.random.uniform(value_min, value_max)
                    matrix_array[n][j][i] = value
                    matrix_array[n][i][j] = value


def add_blocks(
        matrix_array: np.ndarray,
        block_starts: Union[np.ndarray, None],
        density_array: np.ndarray,
        density_min: float,
        density_max: float,
        value_min: float,
        value_max: float,
        block_size_min: int,
        block_size_max: int,
        block_gap_chance: float,
        block_size_average: float,
        block_size_std_dev: float,
) -> List[int]:
    # unpack and validate number of matrices and dimension
    number_of_matrices, dimension, dimension_check = matrix_array.shape
    if dimension_check != dimension:
        raise ValueError("Dimension mismatch")

    # create block size generator for truncated bell curve
    scale = int(block_size_average * block_size_std_dev)
    lower_bound_offset = (block_size_min - block_size_average) / scale
    upper_bound_offset = (block_size_max - block_size_average) / scale
    size_generator = stats.truncnorm(lower_bound_offset, upper_bound_offset, loc=block_size_average, scale=scale)

    # create blocks
    size_collector = []
    for n in range(number_of_matrices):
        # generate density for current matrix and add to metadata
        block_density: float = np.random.uniform(density_min, density_max)
        density_array[n] = block_density
        index = 0
        while index < dimension - 1:
            # add random gap depending on gap chance
            if np.random.uniform(0.0, 1.0) < block_gap_chance:
                index += 1
            else:
                if block_starts is not None:
                    block_starts[n][index] = 1.0
                current_block_size = int(size_generator.rvs())

                # guard against leaving a single element (instead expand current_block_size)
                if dimension - (current_block_size + index) < block_size_min:
                    current_block_size = dimension - index - 1

                # guard against overshooting the matrix size
                if current_block_size + index >= dimension:
                    current_block_size = dimension - index
                    if current_block_size < block_size_max:
                        raise ValueError("Clamped block size is too small")

                for j in range(current_block_size):
                    a = j + index
                    # set diagonal to value
                    if np.random.random() < block_density:
                        matrix_array[n][a][a] = np.random.uniform(value_min, value_max)
                    for i in range(j):
                        b = i + index
                        if np.random.random() < block_density:
                            value = np.random.uniform(value_min, value_max)
                            matrix_array[n][a][b] = value
                            matrix_array[n][b][a] = value
                index += current_block_size
                # collect size for histogram creation
                size_collector.append(current_block_size)

    return size_collector
