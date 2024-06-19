# encoding: utf-8

import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


class ValueProperties:
    den_min: float
    den_max: float
    val_min: float
    val_max: float

    def __init__(self, density_range: tuple[float, float], value_range: tuple[float, float]):
        self.den_min, self.den_max = density_range
        self.val_min, self.val_max = value_range


class BlockProperties:
    len_min: int
    len_max: int
    len_avg: float
    len_sdv: float
    gap_chn: float

    def __init__(self, size_range: tuple[int, int], size_average: float, size_std_dev: float, gap_chance: float):
        self.len_min, self.len_max = size_range
        self.len_avg = size_average
        self.len_sdv = size_std_dev
        self.gap_chn = gap_chance


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

    # parameters
    bgr_noise_vp: ValueProperties
    blk_noise_vp: ValueProperties
    blk_noise_bp: BlockProperties
    blk_tdata_vp: ValueProperties
    blk_tdata_bp: BlockProperties

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
            background_noise_value_properties: ValueProperties,
            block_noise_value_properties: ValueProperties,
            block_noise_block_properties: BlockProperties,
            block_data_value_properties: ValueProperties,
            block_data_block_properties: BlockProperties,
            seed: int = None,
            determinant_cutoff: float = 0.0,
            print_debug: bool = False,
            band_padding_value: np.float32 = np.NAN,
    ):
        self.dim = dimension
        self.len = sample_size
        self.band_rad = band_radius
        # background noise parameters
        self.bgr_noise_vp = background_noise_value_properties
        # block noise parameters
        self.blk_noise_vp = block_noise_value_properties
        self.blk_noise_bp = block_noise_block_properties
        # block true data parameters
        self.blk_tdata_vp = block_data_value_properties
        self.blk_tdata_bp = block_data_block_properties
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
                bins=list(range(self.blk_noise_bp.len_min, self.blk_noise_bp.len_max + 1)),
                alpha=0.5,
                label='Noise'
            )
            plt.hist(
                self.block_data_sizes,
                bins=list(range(self.blk_tdata_bp.len_min, self.blk_tdata_bp.len_max + 1)),
                alpha=0.5,
                label='Data'
            )
            plt.legend(loc='upper right')
            plt.show()

    def __add_background_noise(self, i: int) -> None:
        # create some random noise values (some might be overridden later)
        noise_density = np.random.uniform(self.bgr_noise_vp.den_min, self.bgr_noise_vp.den_max)
        self.metadata[i].bgr_noise_den = noise_density  # store generated noise density in metadata field

        # initialize truth-matrix (selector)
        sel: np.ndarray = np.zeros((self.dim, self.dim), dtype=bool)
        # populate lower triangular matrix selector
        size: int = ((self.dim * (self.dim + 1)) // 2)
        sel[np.tril_indices(self.dim)] = np.random.uniform(size=size) < noise_density
        # add values to matrix's lower triangular based on selector
        self.matrices[i][sel] = np.random.uniform(self.bgr_noise_vp.val_min, self.bgr_noise_vp.val_max, size=sel.sum())
        # copy lower triangular back onto upper triangular via transpose
        self.matrices[i][np.triu_indices(self.dim)] = self.matrices[i].T[np.triu_indices(self.dim)]
        # check if symmetrical
        self.metadata[i].det = np.allclose(self.matrices[i], self.matrices[i].T, rtol=1e-05, atol=1e-08)

    def __add_blocks(self, generate_type: str) -> list[int]:
        if generate_type == "noise":
            value_properties: ValueProperties = self.blk_noise_vp
            block_properties: BlockProperties = self.blk_noise_bp
            start_vec: np.ndarray = self.block_noise_start_labels
            generate_true_data: bool = False
        elif generate_type == "data":
            value_properties: ValueProperties = self.blk_tdata_vp
            block_properties: BlockProperties = self.blk_tdata_bp
            start_vec: np.ndarray = self.block_data_start_labels
            generate_true_data: bool = True
        else:
            raise ValueError(f"generate_type {generate_type} not supported")

        # create block size generator for truncated bell curve
        scale: int = int(block_properties.len_avg * block_properties.len_sdv)
        lbo: float = (block_properties.len_min - block_properties.len_avg) / scale  # lower bound offset
        ubo: float = (block_properties.len_max - block_properties.len_avg) / scale  # upper bound offset
        size_generator: stats.rv_continuous = stats.truncnorm(a=lbo, b=ubo, loc=block_properties.len_avg, scale=scale)

        # debug counter
        generated_counter: int = 0

        # create blocks
        size_collector: list[int] = []
        for index in range(self.len):
            # generate density for current matrix and add to metadata
            block_density: float = np.random.uniform(value_properties.den_min, value_properties.den_max)
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
                    value_properties,
                    block_properties,
                    start_vec,
                    block_density
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
        value_properties: ValueProperties,
        block_properties: BlockProperties,
        start_vec: np.ndarray,
        block_density: float,
    ) -> bool:
        row_index = 0
        while row_index < self.dim - 1:
            start_vec[matrix_index][row_index] = 0  # initialize the value
            # add random gap depending on gap chance
            draw: float = np.random.uniform(0.0, 1.0)
            if draw < block_properties.gap_chn:
                start_vec[matrix_index][row_index] = -1  # denote end of block if gaps are allowed
                row_index += 1
            else:
                start_vec[matrix_index][row_index] = 1
                current_block_size: int = int(size_generator.rvs())

                # guard against leaving a single element (instead expand current_block_size)
                if self.dim - (current_block_size + row_index) < block_properties.len_min:
                    current_block_size = self.dim - row_index - 1

                # guard against overshooting the matrix size
                if current_block_size + row_index >= self.dim:
                    current_block_size = self.dim - row_index
                    if current_block_size < block_properties.len_max:
                        raise ValueError("Clamped block size is too small")

                for j in range(current_block_size):
                    a = j + row_index
                    # set diagonal to value
                    if np.random.random() < block_density:
                        value: float = np.random.uniform(value_properties.val_min, value_properties.val_max)
                        self.matrices[matrix_index][a][a] = np.float32(value)
                    for i in range(j):
                        b = i + row_index
                        if np.random.random() < block_density:
                            value: float = np.random.uniform(value_properties.val_min, value_properties.val_max)
                            self.matrices[matrix_index][a][b] = np.float32(value)
                            self.matrices[matrix_index][b][a] = np.float32(value)
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
    bgr_noise_value_props = ValueProperties(density_range=(0.3, 0.5), value_range=(0.0, 0.5))
    noise_blk_value_props = ValueProperties(density_range=(0.3, 0.5), value_range=(0.3, 1.0))
    noise_blk_block_props = BlockProperties(size_range=(3, 32), size_average=10, size_std_dev=0.66, gap_chance=0.5)
    tdata_blk_value_props = ValueProperties(density_range=(0.5, 0.7), value_range=(0.3, 1.0))
    tdata_blk_block_props = BlockProperties(size_range=(2, 32), size_average=10, size_std_dev=0.66, gap_chance=0.5)

    test_data = MatrixData(
        dimension=64,
        band_radius=10,
        sample_size=1000,
        background_noise_value_properties=bgr_noise_value_props,
        block_noise_value_properties=noise_blk_value_props,
        block_noise_block_properties=noise_blk_block_props,
        block_data_value_properties=tdata_blk_value_props,
        block_data_block_properties=tdata_blk_block_props,
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
            test_data.bands[selected_index],
            cmap='rocket',
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            square=True,
            vmin=0,
            vmax=1
        )

        data_fig.show()
