from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

import matrixlib.util
from matrixlib.metadata import MatrixMetadata


# define color bars and tick labels (e.g. 'rocket', 'rocket_r', 'viridis', 'flare', 'magma' ...)
VALUE_COLORBAR = 'rocket'
BLOCK_COLORBAR = 'flare'


def generate_block_matrix(matrix_block_start_vector: np.ndarray) -> np.array:
    dimension = matrix_block_start_vector.shape[0]
    block_matrix_array = np.zeros((dimension, dimension), dtype=float)
    block_start_array = matrix_block_start_vector

    index = 0
    counter = 0
    for k in range(dimension):
        if block_start_array[k]:
            # reset block counters
            index += counter
            counter = 0

        counter += 1

        if (k + 1) == dimension:  # i.e. last entry on the diagonal
            draw_now = True
        else:
            draw_now = block_start_array[k + 1]

        if draw_now:
            for i in range(counter):
                for j in range(counter):
                    block_matrix_array[index + i][index + j] = 1
    return block_matrix_array


def plot_matrices_and_metadata(
        figure: plt.Figure,
        shape: (int, int),
        matrix_indices: List[int],
        matrix_data: np.ndarray,
        matrix_metadata: MatrixMetadata,
) -> None:
    cbar_map_values = VALUE_COLORBAR
    cbar_map_blocks = BLOCK_COLORBAR
    cbar_kws = {'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}

    _, dim, _ = matrix_data.shape
    num_of_subplots = len(matrix_indices)
    row_col_number = shape[0] * 100 + shape[1] * 10

    for i in range(num_of_subplots):
        this_index = matrix_indices[i]
        this_hex_str = matrixlib.util.generate_block_vector_hex_string(matrix_metadata.block_starts[this_index])

        sp1 = figure.add_subplot(row_col_number + 2 * i + 1)
        sp1.set_title(f"Matrix [{this_index}] values ({this_hex_str})")
        sns.heatmap(
            matrix_data[this_index],
            cmap=cbar_map_values,
            cbar_kws=cbar_kws,
            xticklabels=False,
            yticklabels=False,
            square=True,
            vmin=0,
            vmax=1
        )

        sp2 = figure.add_subplot(row_col_number + 2 * i + 2)
        sp2.set_title(f"Matrix [{this_index}] blocks")
        sns.heatmap(
            generate_block_matrix(matrix_metadata.block_starts[this_index]),
            cmap=cbar_map_blocks,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            square=True
        )
