from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from . import core, util

# define color bars and tick labels (e.g. 'rocket', 'rocket_r', 'viridis', 'flare', 'magma' ...)
VALUE_COLORBAR = 'rocket'
BLOCK_COLORBAR = 'flare'


def generate_block_matrix(matrix_block_start_vector: np.ndarray) -> np.array:
    print(f"matrix block start -> {matrix_block_start_vector}")
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
        fig_shape: (int, int),
        matrix_indices: list[int],
        matrix_data: core.MatrixData,
) -> None:
    cbar_map_values = VALUE_COLORBAR
    cbar_map_blocks = BLOCK_COLORBAR
    cbar_kws = {'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}

    num_of_subplots = len(matrix_indices)
    row_col_number = fig_shape[0] * 100 + fig_shape[1] * 10

    for i in range(num_of_subplots):
        this_index = matrix_indices[i]
        this_hex_str = util.generate_block_vector_hex_string(matrix_data.noise_blk_starts[this_index])

        sp1 = figure.add_subplot(row_col_number + 2 * i + 1)
        sp1.set_title(f"Matrix [{this_index}] values ({this_hex_str})")
        sns.heatmap(
            matrix_data.matrices[this_index],
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
            generate_block_matrix(matrix_data.noise_blk_starts[this_index]),
            cmap=cbar_map_blocks,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            square=True
        )
