# encoding: utf-8

import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


class BlockProperties:
    blk_len_avg: float
    blk_len_sdv: float
    blk_len_min: int
    blk_len_max: int
    blk_den_min: float
    blk_den_max: float
    blk_gap_chn: float
    blk_val_min: np.float32
    blk_val_max: np.float32
    blk_start_v: np.ndarray

    def __init__(
        self,
        size_average: float,
        size_std_dev: float,
        size_min: int,
        size_max: int,
        density_min: float,
        density_max: float,
        gap_chance: float,
        value_min: np.float32,
        value_max: np.float32,
    ):
        self.blk_len_avg = size_average
        self.blk_len_sdv = size_std_dev
        self.blk_len_min = size_min
        self.blk_len_max = size_max
        self.blk_den_min = density_min
        self.blk_den_max = density_max
        self.blk_gap_chn = gap_chance
        self.blk_val_min = value_min
        self.blk_val_max = value_max


class MetaData:
    """A data class containing the randomized values that were assigned to the corresponding matrix with the same
    index.

    **Note:** To keep variable names reasonably short the following abbreviations are used:
        | ``bgr_...`` for background
        | ``blk_...`` for blocks
        | also note that ``tdata`` here refers to the `true` data values
    """
    bgr_noise_den: float = None
    bgr_noise_min: float = None
    bgr_noise_max: float = None
    blk_noise_den: float = None
    blk_noise_min: float = None
    blk_noise_max: float = None
    blk_tdata_den: float = None
    blk_tdata_min: float = None
    blk_tdata_max: float = None
    det: float = None

    def __init__(self):
        pass


class MatrixData:
    # provided input
    len: int
    band_rad: int
    dim: int
    determinant_cutoff: float
    band_padding_value: np.float32
    # background parameters
    bgr_noise_den_min: float
    bgr_noise_den_max: float
    bgr_noise_val_min: np.float32
    bgr_noise_val_max: np.float32
    # noise block parameters
    blk_noise_len_min: int
    blk_noise_len_max: int
    blk_noise_len_avg: float
    blk_noise_len_sdv: float
    blk_noise_gap_chn: float
    blk_noise_den_min: float
    blk_noise_den_max: float
    blk_noise_val_min: np.float32
    blk_noise_val_max: np.float32
    # true data block parameters
    blk_tdata_len_min: int
    blk_tdata_len_max: int
    blk_tdata_len_avg: float
    blk_tdata_len_sdv: float
    blk_tdata_gap_chn: float
    blk_tdata_den_min: float
    blk_tdata_den_max: float
    blk_tdata_val_min: np.float32
    blk_tdata_val_max: np.float32

    # debug data
    seed: int
    debug: bool

    # generated output
    matrices: np.ndarray = None
    bands: np.ndarray = None
    block_data_start_labels: np.ndarray = None
    block_noise_start_labels: np.ndarray = None
    metadata: list[MetaData] = None

    # statistics data
    block_noise_sizes: list[int] = None
    block_data_sizes: list[int] = None

    def __init__(
            self,
            dimension: int,
            band_radius: int,
            sample_size: int,
            background_noise_density_range: tuple[float, float],
            background_noise_value_range: (np.float32, np.float32),
            block_noise_density_range: tuple[float, float],
            block_noise_value_range: (np.float32, np.float32),
            block_noise_size_range: tuple[int, int],
            block_noise_size_average: float,
            block_noise_size_std_dev: float,
            block_noise_size_gap_chance: float,
            block_data_density_range: tuple[float, float],
            block_data_value_range: (np.float32, np.float32),
            block_data_size_range: tuple[int, int],
            block_data_size_average: float,
            block_data_size_std_dev: float,
            block_data_size_gap_chance: float,
            seed: int = None,
            determinant_cutoff: float = 0.0,
            print_debug: bool = False,
            band_padding_value: np.float32 = np.NAN,
    ):
        self.dim = dimension
        self.len = sample_size
        self.band_rad = band_radius
        # background noise parameters
        self.bgr_noise_den_min, self.bgr_noise_den_max = background_noise_density_range
        self.bgr_noise_val_min, self.bgr_noise_val_max = background_noise_value_range
        # block noise parameters
        self.blk_noise_den_min, self.blk_noise_den_max = block_noise_density_range
        self.blk_noise_val_min, self.blk_noise_val_max = block_noise_value_range
        self.blk_noise_len_min, self.blk_noise_len_max = block_noise_size_range
        self.blk_noise_len_avg = block_noise_size_average
        self.blk_noise_len_sdv = block_noise_size_std_dev
        self.blk_noise_gap_chn = block_noise_size_gap_chance
        # block true data parameters
        self.blk_tdata_den_min, self.blk_tdata_den_max = block_data_density_range
        self.blk_tdata_val_min, self.blk_tdata_val_max = block_data_value_range
        self.blk_tdata_len_min, self.blk_tdata_len_max = block_data_size_range
        self.blk_tdata_len_avg = block_data_size_average
        self.blk_tdata_len_sdv = block_data_size_std_dev
        self.blk_tdata_gap_chn = block_data_size_gap_chance
        # flags and values used during generation
        self.band_padding_value = band_padding_value
        self.determinant_cutoff = determinant_cutoff
        self.seed = seed
        self.debug = print_debug

        # initialize data arrays and generate matrix data
        self.__init_data_size()
        self.__generate_matrices()
        self.__narrow_to_band()

    def __init_data_size(self) -> None:
        n: int = self.len
        dim: int = self.dim
        r: int = self.band_rad
        w: int = 2 * r + 1  # the bands width is diagonal plus band radius in each direction

        self.matrices = np.zeros(shape=(n, dim, dim), dtype=np.float32)
        self.block_data_start_labels = np.zeros(shape=(n, dim), dtype=np.int8)
        self.block_noise_start_labels = np.zeros(shape=(n, dim), dtype=np.int8)
        self.bands = np.zeros(shape=(n, w, dim), dtype=np.float32)

        self.metadata = [MetaData() for _ in range(n)]

        if self.debug:
            bytes_per_mib = 1024 * 1024
            print(f"initialized        data vectors of size {n:6d} x {dim:3d} x {dim:3d} = {n * dim * dim:9d} " +
                  f"with a memory usage of {self.matrices.nbytes / bytes_per_mib:7.3f} MiB")
            print(f"initialized  data start vectors of size {n:6d} x {dim:3d}       = {n*dim:9d} " +
                  f"with a memory usage of {self.block_data_start_labels.nbytes / bytes_per_mib:7.3f} MiB")
            print(f"initialized noise start vectors of size {n:6d} x {dim:3d}       = {n * dim:9d} " +
                  f"with a memory usage of {self.block_noise_start_labels.nbytes / bytes_per_mib:7.3f} MiB")
            print(f"initialized        band vectors of size {n:6d} x {dim:3d} x {r:3d} = {n * dim * r:9d} " +
                  f"with a memory usage of {self.bands.nbytes / bytes_per_mib:7.3f} MiB")

        return

    def __generate_matrices(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.debug:
            print("generating matrices...")
            print("    ...adding background noise")
        for n in range(self.len):
            self.__add_background_noise(n)
        if self.debug:
            print("    ...adding noise blocks")
        self.block_noise_sizes = self.__add_blocks(generate_type="noise")
        if self.debug:
            print("    ...adding data blocks")
        self.block_data_sizes = self.__add_blocks(generate_type="data")

        if self.debug:
            # make a histogram for the block sizes
            plt.hist(
                self.block_noise_sizes,
                bins=list(range(self.blk_noise_len_min, self.blk_noise_len_max + 1)),
                alpha=0.5,
                label='Noise'
            )
            plt.hist(
                self.block_data_sizes,
                bins=list(range(self.blk_tdata_len_min, self.blk_tdata_len_max + 1)),
                alpha=0.5,
                label='Data'
            )
            plt.legend(loc='upper right')
            plt.show()

    def __add_background_noise(self, i: int) -> None:
        # create some random noise values (some might be overridden later)
        noise_density = np.random.uniform(self.bgr_noise_den_min, self.bgr_noise_den_max)
        self.metadata[i].bgr_noise_den = noise_density  # store generated noise density in metadata field

        # initialize truth-matrix (selector)
        sel: np.ndarray = np.zeros((self.dim, self.dim), dtype=bool)
        # populate lower triangular matrix selector
        size: int = ((self.dim * (self.dim + 1)) // 2)
        sel[np.tril_indices(self.dim)] = np.random.uniform(size=size) < noise_density
        # add values to matrix's lower triangular based on selector
        self.matrices[i][sel] = np.random.uniform(self.bgr_noise_val_min, self.bgr_noise_val_max, size=sel.sum())
        # copy lower triangular back onto upper triangular via transpose
        self.matrices[i][np.triu_indices(self.dim)] = self.matrices[i].T[np.triu_indices(self.dim)]
        # check if symmetrical
        self.metadata[i].det = np.allclose(self.matrices[i], self.matrices[i].T, rtol=1e-05, atol=1e-08)

    def __add_blocks(self, generate_type: str) -> list[int]:
        if generate_type == "noise":
            generate_true_data = False
        elif generate_type == "data":
            generate_true_data = True
        else:
            raise ValueError(f"generate_type {generate_type} not supported")

        # select proper block type
        block_size_average: float = self.blk_tdata_len_avg if generate_true_data else self.blk_noise_len_avg
        block_size_std_dev: float = self.blk_tdata_len_sdv if generate_true_data else self.blk_noise_len_sdv
        block_size_min: int = self.blk_tdata_len_min if generate_true_data else self.blk_noise_len_min
        block_size_max: int = self.blk_tdata_len_max if generate_true_data else self.blk_noise_len_max
        density_min: float = self.blk_tdata_den_min if generate_true_data else self.blk_noise_den_min
        density_max: float = self.blk_tdata_den_max if generate_true_data else self.blk_noise_den_max
        block_gap_chance: float = self.blk_tdata_gap_chn if generate_true_data else self.blk_noise_gap_chn
        block_starts: np.ndarray = self.block_data_start_labels if generate_true_data else self.block_noise_start_labels
        value_min: np.float32 = self.blk_tdata_val_min if generate_true_data else self.blk_noise_val_min
        value_max: np.float32 = self.blk_tdata_val_max if generate_true_data else self.blk_noise_val_max

        # create block size generator for truncated bell curve
        scale = int(block_size_average * block_size_std_dev)
        lower_bound_offset = (block_size_min - block_size_average) / scale
        upper_bound_offset = (block_size_max - block_size_average) / scale
        size_generator = stats.truncnorm(lower_bound_offset, upper_bound_offset, loc=block_size_average, scale=scale)

        # debug counter
        generated_counter: int = 0

        # create blocks
        size_collector: list[int] = []
        for index in range(self.len):
            # generate density for current matrix and add to metadata
            block_density: float = np.random.uniform(density_min, density_max)
            if generate_true_data:
                self.metadata[index].blk_tdata_den = block_density
            else:
                self.metadata[index].blk_noise_den = block_density

            matrix_invalid: bool = True
            while matrix_invalid:
                matrix_invalid = self.__generate_single_matrix(
                    index,
                    size_collector,
                    size_generator,
                    block_starts,
                    block_size_min,
                    block_size_max,
                    block_density,
                    block_gap_chance,
                    value_min,
                    value_max,
                )
                generated_counter += 1

        if self.debug:
            invalid_matrices = generated_counter - self.len
            print(f"Generated a total of {generated_counter} matrices since {invalid_matrices} were invalid.")

        return size_collector

    def __generate_single_matrix(
        self,
        matrix_index: int,
        size_collector: list[int],
        size_generator: stats.truncnorm,
        block_starts: np.ndarray,
        block_size_min: int,
        block_size_max: int,
        block_density: float,
        block_gap_chance: float,
        val_min: np.float32,
        val_max: np.float32,
    ) -> bool:
        row_index = 0
        while row_index < self.dim - 1:
            block_starts[matrix_index][row_index] = 0  # initialize the value
            # add random gap depending on gap chance
            draw: float = np.random.uniform(0.0, 1.0)
            if draw < block_gap_chance:
                block_starts[matrix_index][row_index] = -1  # denote end of block if gaps are allowed
                row_index += 1
            else:
                block_starts[matrix_index][row_index] = 1
                current_block_size: int = int(size_generator.rvs())

                # guard against leaving a single element (instead expand current_block_size)
                if self.dim - (current_block_size + row_index) < block_size_min:
                    current_block_size = self.dim - row_index - 1

                # guard against overshooting the matrix size
                if current_block_size + row_index >= self.dim:
                    current_block_size = self.dim - row_index
                    if current_block_size < block_size_max:
                        raise ValueError("Clamped block size is too small")

                for j in range(current_block_size):
                    a = j + row_index
                    # set diagonal to value
                    if np.random.random() < block_density:
                        self.matrices[matrix_index][a][a] = np.random.uniform(val_min, val_max)
                    for i in range(j):
                        b = i + row_index
                        if np.random.random() < block_density:
                            value = np.random.uniform(val_min, val_max)
                            self.matrices[matrix_index][a][b] = value
                            self.matrices[matrix_index][b][a] = value
                row_index += current_block_size
                # collect size for histogram creation
                size_collector.append(current_block_size)

        det: float = np.linalg.det(self.matrices[matrix_index])
        matrix_invalid: bool = abs(det) < self.determinant_cutoff
        if matrix_invalid and self.debug:
            print(f"determinant {det} of [{matrix_index}] is below cutoff threshold, recalculating.")
        return matrix_invalid

    def __narrow_to_band(self) -> None:
        for k in range(self.len):
            # be wary of cache effects here!
            for j in range(self.dim):
                self.bands[k][self.band_rad][j] = self.matrices[k][j][j]  # process the diagonal
                for i in range(self.band_rad):
                    o = i - self.band_rad
                    u = 2 * self.band_rad - i
                    if j > self.band_rad - i - 1:
                        self.bands[k][i][j] = self.matrices[k][j][j + o]
                        self.bands[k][u][j] = self.matrices[k][j][j + o]
                    else:
                        # use nan for better plotting, might be necessary to pad to 0 for training
                        self.bands[k][i][j] = self.band_padding_value
                        self.bands[k][u][j] = self.band_padding_value
        return


if __name__ == "__main__":
    test_data = MatrixData(
        dimension=64,
        band_radius=10,
        sample_size=1000,
        background_noise_density_range=(0.3, 0.5),
        background_noise_value_range=(0.0, 0.5),
        block_noise_density_range=(0.3, 0.5),
        block_noise_value_range=(0.3, 1.0),
        block_noise_size_range=(3, 32),
        block_noise_size_average=10,
        block_noise_size_std_dev=0.66,
        block_noise_size_gap_chance=0.5,
        block_data_density_range=(0.5, 0.7),
        block_data_value_range=(0.3, 1.0),
        block_data_size_range=(2, 32),
        block_data_size_average=10,
        block_data_size_std_dev=0.66,
        block_data_size_gap_chance=0.0,
        seed=42,
        determinant_cutoff=0.01,
        print_debug=True
    )

    for selected_index in range(4):
        data_fig = plt.figure(num=selected_index, figsize=(14, 5))
        data_fig.suptitle(f"test [{selected_index}] - {test_data.metadata[selected_index].bgr_noise_den}")
        # plot the matrix
        sp1 = data_fig.add_subplot(1, 2, 1)
        sns.heatmap(
            test_data.matrices[selected_index],
            cmap='rocket',
            cbar_kws={'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            xticklabels=False,
            yticklabels=False,
            square=True,
            vmin=0,
            vmax=1
        )
        # plot the band
        sp2 = data_fig.add_subplot(1, 2, 2)
        sns.heatmap(
            test_data.bands[0],
            cmap='rocket',
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            square=True,
            vmin=0,
            vmax=1
        )

        data_fig.show()
