import numpy as np
from PIL import Image
from . import util


def write_matrix_to_file(
        index: int,
        base_path: str,
        matrix_data: np.ndarray,
        block_starts: np.ndarray,
        noise_background_density: float,
        noise_block_density: float,
        true_block_density: float,
        title: str = None,
) -> None:
    rows, cols = matrix_data.shape
    int_vector = np.zeros((rows, cols, 4), np.uint8)
    for c in range(cols):
        for e in range(rows):
            # projecting interval [0, 1] in real numbers onto [0, 4294967295] in natural numbers
            int_value = 0 if np.isnan(matrix_data[e][c]) else int(matrix_data[e][c] * np.iinfo(np.uint32).max)
            int_vector[e][c][0] = np.uint8((int_value & 0b11111111000000000000000000000000) >> 24)
            int_vector[e][c][1] = np.uint8((int_value & 0b00000000111111110000000000000000) >> 16)
            int_vector[e][c][2] = np.uint8((int_value & 0b00000000000000001111111100000000) >> 8)
            int_vector[e][c][3] = np.uint8((int_value & 0b00000000000000000000000011111111))

    img = Image.fromarray(int_vector)  # magic number is max(int32)

    # get title from input or generate from metadata
    if title is not None:
        file_name = title
    else:
        block_start_hex_str = util.generate_block_vector_hex_string(block_starts)
        file_name = (
            f"{index:04d}-" +
            f"{noise_background_density:0.3f}-" +
            f"{noise_block_density:0.3f}-" +
            f"{true_block_density:0.3f}-" +
            f"{block_start_hex_str}"
        )
    img.save(f"{base_path}/data/{file_name}.png", "PNG")


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

                data_vector[e][c] = (np.float32(int_value) / np.float32(np.iinfo(np.uint32).max))
        return data_vector
    except Exception as e:
        raise RuntimeError(f"Error loading and preprocessing image {file_path}: {e}")
