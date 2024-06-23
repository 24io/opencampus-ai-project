import numpy as np


def generate_block_vector_hex_string(block_start_vector: np.array) -> str:
    """Creates a hex string representing the ones and zeroes of a given block start vector.

    :param block_start_vector: A ``np.ndarray`` indicating the starts of blocks in an associated matrix.
    :return: A ``str`` of the hex representation of the block start vector.
    """
    base = 1  # use this base 'counter' to prevent call of pow()
    int_rep = 0
    for e in block_start_vector:
        if e != 0:
            int_rep += base
        base *= 2
    return f"{int_rep:016x}"


def apply_minmax_norm(matrices: np.array, factor: float = 1.0, offset: float = 0.0) -> None:
    """**Mutates** the given ``matrices`` applying min-max normalization with optional ``factor`` and ``offset``.

    The ``factor`` shifts the interval top (1) and resulting in the target interval [0, factor] or [factor, 0] of the
    provided ``factor`` is negative.

    The ``offset`` subsequently shifts the interval scaled by ``factor``.

    **Note:** Both ``factor`` and ``offset`` influence the **top** of the final interval.

    :param matrices: An ``np.ndarray`` of symmetric matrices.
    :param factor: A ``float`` factor to be applied to the min-max normalization.
    :param offset: A ``float`` offset to be applied to the min-max normalization.
    """
    num_of_matrices: int
    dim_of_matrices: int
    num_of_matrices, dim_of_matrices, _ = matrices.shape

    for i in range(num_of_matrices):
        val_min, val_max = matrices[i].min(), matrices[i].max()
        # normalize the matrix values to interval
        matrices[i] = (factor * ((matrices[i] - val_min) / (val_max - val_min))) + offset
