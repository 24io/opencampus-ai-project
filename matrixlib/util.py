import numpy as np
import tensorflow as tf


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


def narrow_to_band(data: np.ndarray, radius: int) -> np.ndarray:
    entries, rows, _ = data.shape
    band_width = 2 * radius + 1
    result = np.zeros((entries, band_width, rows))
    for k in range(entries):
        # be wary of cache effects here!
        for j in range(rows):
            result[k][radius][j] = data[k][j][j]  # process the diagonal
            for i in range(radius):
                o = i - radius
                u = band_width - i - 1
                if j > radius - i - 1:
                    result[k][i][j] = data[k][j][j + o]
                    result[k][u][j] = data[k][j][j + o]
                else:
                    # use nan for better plotting, might be necessary to pad to 0 for training
                    result[k][i][j] = np.NAN
                    result[k][u][j] = np.NAN
    return result


def to_tensorflow_dataset(matrix_data: np.ndarray, matrix_labels: np.ndarray) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((matrix_data, matrix_labels))
