import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from . import core, util

# define color bars and tick labels (e.g. 'rocket', 'rocket_r', 'viridis', 'flare', 'magma' ...)
VALUE_COLORBAR = 'rocket'
BLOCK_COLORBAR = 'flare'
SUBPLOTS: int = 4


def generate_block_matrix(matrix_block_start_vector: np.ndarray, debug: bool = False) -> np.array:
    if debug:
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
            draw_now = block_start_array[k + 1] and not block_start_array[k] == -1

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
    cols, rows = fig_shape
    cbar_map_values = VALUE_COLORBAR
    cbar_map_blocks = BLOCK_COLORBAR
    cbar_kws = {'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}

    num_of_subplots = len(matrix_indices)
    row_col_number = cols * 100 + rows * 10

    figure.patch.set_alpha(0.0)  # make background transparent

    for i in range(num_of_subplots):
        this_index = matrix_indices[i]
        this_hex_str = util.generate_block_vector_hex_string(matrix_data.noise_blk_starts[this_index])

        sp1 = figure.add_subplot(cols, rows, SUBPLOTS * i + 1)
        sp1.set_title(f"Matrix [{this_index}] diagonal band\nstart-vector-hex: {this_hex_str}")
        sns.heatmap(
            matrix_data.bands[this_index],
            cmap=cbar_map_values,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            square=True
        )
        sp1.patch.set_alpha(0.0)  # make subplot 1 transparent

        sp2 = figure.add_subplot(cols, rows, SUBPLOTS * i + 2)
        sp2.set_title(f"Matrix [{this_index}] values")
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
        sp2.patch.set_alpha(0.0)  # make subplot 2 transparent

        sp3 = figure.add_subplot(cols, rows, SUBPLOTS * i + 3)
        sp3.set_title(f"Matrix [{this_index}] noise blocks")
        sns.heatmap(
            generate_block_matrix(matrix_data.noise_blk_starts[this_index]),
            cmap=cbar_map_blocks,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            square=True
        )

        sp3.patch.set_alpha(0.0)  # make subplot 3 transparent

        sp4 = figure.add_subplot(cols, rows, SUBPLOTS * i + 4)
        sp4.set_title(f"Matrix [{this_index}] data blocks")
        sns.heatmap(
            generate_block_matrix(matrix_data.tdata_blk_starts[this_index]),
            cmap=cbar_map_blocks,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            square=True
        )

        sp4.patch.set_alpha(0.0)  # make subplot 4 transparent
