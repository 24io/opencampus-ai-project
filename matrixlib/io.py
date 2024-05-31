import numpy as np
from PIL import Image
from matrixlib.util import generate_block_vector_hex_string


def write_matrix_to_file(
        index: int,
        base_path: str,
        matrix_data: np.ndarray,
        d_noise: float,
        d_block: float,
        s_block: np.array,
) -> None:
    cols, rows = matrix_data.shape
    block_start_hex_str = generate_block_vector_hex_string(s_block)
    int_vector = np.zeros((rows, cols, 4), np.uint8)
    for c in range(cols):
        for e in range(rows):
            # projecting interval [0, 1] in real numbers onto [0, 4294967295] in natural numbers
            int_value = int(matrix_data[e][c] * 4294967295)  # = max(uint32) = 2**32 - 1

            int_vector[e][c][0] = np.uint8((int_value & 0b11111111000000000000000000000000) >> 24)
            int_vector[e][c][1] = np.uint8((int_value & 0b00000000111111110000000000000000) >> 16)
            int_vector[e][c][2] = np.uint8((int_value & 0b00000000000000001111111100000000) >> 8)
            int_vector[e][c][3] = np.uint8((int_value & 0b00000000000000000000000011111111))

    img = Image.fromarray(int_vector)  # magic number is max(int32)
    img.save(f"{base_path}/data/{index:04d}-{d_noise:0.3f}-{d_block:0.3f}-{block_start_hex_str}.png", "PNG")


def read_matrix_from_file(file_path: str) -> np.ndarray:
    try:
        image_vector = np.array(Image.open(file_path))
        rows, cols, _ = image_vector.shape
        data_vector = np.zeros((rows, cols), np.float32)

        for c in range(cols):
            for e in range(rows):
                int_value: np.uint32 = np.uint32(0)

                int_value += (image_vector[e][c][0] << 24) & 0b11111111000000000000000000000000
                int_value += (image_vector[e][c][1] << 16) & 0b00000000111111110000000000000000
                int_value += (image_vector[e][c][2] << 8) & 0b00000000000000001111111100000000
                int_value += (image_vector[e][c][3])

                data_vector[e][c] = (np.float32(int_value) / 4294967295.0)
        return data_vector
    except Exception as e:
        raise RuntimeError(f"Error loading and preprocessing image {file_path}: {e}")
