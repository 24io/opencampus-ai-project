import numpy as np


def generate_block_vector_hex_string(block_vector: np.array) -> str:
    base = 1  # use this base 'counter' to prevent call of pow()
    int_rep = 0
    for e in block_vector:
        if e:
            int_rep += base
        base *= 2
    return f"{int_rep:016x}"


def shift_normalize(matrix: np.array) -> np.array:
    min_val: float = np.min(matrix)
    max_val: float = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)
