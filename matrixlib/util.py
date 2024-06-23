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


def apply_minmax_norm(matrices: np.array, factor: float = 1.0, offset: float = 0.0) -> np.ndarray:
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

    result = np.zeros_like(matrices)
    if len(matrices.shape) == 2:
        num_of_matrices = 1
        # transform to list since we need to access the single matrix as matrices[i]
        matrices = [matrices]
        result = [result]
    elif len(matrices.shape) == 3:
        num_of_matrices, _, _ = matrices.shape
    else:
        raise ValueError("Matrices must be either a 2-D matrix or an array of 2-D matrices")

    for i in range(num_of_matrices):
        val_min, val_max = matrices[i].min(), matrices[i].max()
        result[i] = (factor * ((matrices[i] - val_min) / (val_max - val_min))) + offset

    return np.asarray(result)
