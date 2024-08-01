# encoding: utf-8
import time
import numpy as np
import seaborn as sns

from scipy import stats
from matplotlib import pyplot


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
        if not size_range[0] > 0:
            raise ValueError('size_range lower bound must be greater than zero')
        if not size_range[0] < size_range[1]:
            raise ValueError('lower bounds must be smaller than uppers bound')
        self.len_min, self.len_max = size_range
        self.len_avg = size_average
        self.len_sdv = size_std_dev
        self.gap_chn = gap_chance

    def get_size_generator(self) -> stats.rv_continuous:
        # create block size generator for truncated bell curve
        scale: int = int(self.len_avg * self.len_sdv)
        lower_bound_offset: float = (self.len_min - self.len_avg) / scale
        upper_bound_offset: float = (self.len_max - self.len_avg) / scale
        return stats.truncnorm(a=lower_bound_offset, b=upper_bound_offset, loc=self.len_avg, scale=scale)


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
    force_nonzero_diag: bool
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
    noise_blk_starts: np.ndarray = None
    tdata_blk_starts: np.ndarray = None
    metadata: list[MetaData] = None

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
            force_nonzero_diag: bool = True,
            band_padding_value: np.float32 = 0.0,
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
        self.force_nonzero_diag = force_nonzero_diag
        self.seed = seed
        self.debug = print_debug

        # initialize data arrays and generate matrix data
        start: float = time.time()
        self.__init_data_size()
        self.__generate_matrices()
        self.__narrow_to_band()
        end: float = time.time()
        if self.debug:
            print(f"duration: {end - start:0.2f} seconds")

    def __init_data_size(self) -> None:
        n: int = self.len
        dim: int = self.dim
        r: int = self.band_rad
        w: int = 2 * r + 1  # the bands width is diagonal plus band radius in each direction

        self.matrices = np.zeros(shape=(n, dim, dim), dtype=np.float32)
        self.noise_blk_starts = np.zeros(shape=(n, dim), dtype=np.int8)
        self.tdata_blk_starts = np.zeros(shape=(n, dim), dtype=np.int8)
        self.bands = np.zeros(shape=(n, w, dim), dtype=np.float32)

        self.metadata = [MetaData() for _ in range(n)]

        if self.debug:
            bytes_per_mib = 1024 * 1024
            print(f"initialized        data vectors of size {n:6d} x {dim:3d} x {dim:3d} = {n * dim * dim:9d} " +
                  f"with a memory usage of {self.matrices.nbytes / bytes_per_mib:7.3f} MiB")
            print(f"initialized  data start vectors of size {n:6d} x {dim:3d}       = {n*dim:9d} " +
                  f"with a memory usage of {self.noise_blk_starts.nbytes / bytes_per_mib:7.3f} MiB")
            print(f"initialized noise start vectors of size {n:6d} x {dim:3d}       = {n * dim:9d} " +
                  f"with a memory usage of {self.tdata_blk_starts.nbytes / bytes_per_mib:7.3f} MiB")
            print(f"initialized        band vectors of size {n:6d} x {dim:3d} x {r:3d} = {n * dim * r:9d} " +
                  f"with a memory usage of {self.bands.nbytes / bytes_per_mib:7.3f} MiB")
            print("-" * 80)

        return

    def __generate_matrices(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.debug:
            print("instantiation rng generators...")
        noise_blk_size_gen: stats.rv_continuous = self.blk_noise_bp.get_size_generator()
        tdata_blk_size_gen: stats.rv_continuous = self.blk_tdata_bp.get_size_generator()

        if self.debug:
            print("generating matrices...")
        invalid_counter: int = 0  # counter to geet number of "re-rolls"
        for i in range(self.len):
            matrix_valid: bool = False
            while not matrix_valid:
                self.__add_background_noise(i)
                self.__add_noise_blocks(i, noise_blk_size_gen)
                self.__add_tdata_blocks(i, tdata_blk_size_gen)
                # check if "invertible enough"
                det: float = np.linalg.det(self.matrices[i])
                matrix_valid = abs(det) > self.determinant_cutoff
                self.metadata[i].det = det
                if not matrix_valid:
                    if self.debug:
                        print(f"matrix at [{i}] is invalid (det = {det}) -> re-generating.")
                    invalid_counter += 1

        if self.debug:
            # print info on re-rolled matrices
            print(f"invalid matrices: {invalid_counter}")
            abs_determinants: np.ndarray = np.asarray([abs(self.metadata[i].det) for i in range(self.len)])
            print(f"determinant abs-value range: [{abs_determinants.min()}, {abs_determinants.max()}]")

            # make a histogram for the block sizes
            pyplot.hist(
                self.get_list_of_block_sizes(self.tdata_blk_starts),
                bins=list(range(self.blk_noise_bp.len_min, self.blk_noise_bp.len_max + 1)),
                alpha=0.5,
                label='Noise'
            )
            pyplot.hist(
                self.get_list_of_block_sizes(self.noise_blk_starts),
                bins=list(range(self.blk_tdata_bp.len_min, self.blk_tdata_bp.len_max + 1)),
                alpha=0.5,
                label='Data'
            )
            pyplot.legend(loc='upper right')
            pyplot.show()

    def __add_background_noise(self, i: int) -> None:
        # create some random noise values (some might be overridden later)
        noise_density = np.random.uniform(self.bgr_noise_vp.den_min, self.bgr_noise_vp.den_max)
        # store generated noise density in metadata field
        self.metadata[i].bgr_noise_den = noise_density
        # generate data and write to matrix
        self.matrices[i] = self.__generate_value_space(noise_density, self.bgr_noise_vp)

    def __add_noise_blocks(self, i: int, size_generator: stats.rv_continuous) -> None:
        d: float = self.__add_block(i, size_generator, self.blk_noise_vp, self.blk_noise_bp, self.noise_blk_starts)
        self.metadata[i].blk_noise_den = d

    def __add_tdata_blocks(self, i: int, size_generator: stats.rv_continuous) -> None:
        d: float = self.__add_block(i, size_generator, self.blk_tdata_vp, self.blk_tdata_bp, self.tdata_blk_starts)
        self.metadata[i].blk_tdata_den = d

    def __add_block(
        self,
        mat_index: int,
        size_generator: stats.rv_continuous,
        value_properties: ValueProperties,
        block_properties: BlockProperties,
        start_vec: np.ndarray,
    ) -> float:
        block_density: float = np.random.uniform(value_properties.den_min, value_properties.den_max)
        row_index = 0
        while row_index < self.dim - 1:
            start_vec[mat_index][row_index] = 0  # initialize the value
            # add random gap depending on gap chance
            draw: float = np.random.uniform(0.0, 1.0)
            if draw < block_properties.gap_chn and row_index < self.dim - 1 - block_properties.len_min:
                start_vec[mat_index][row_index] = -1  # denote that index is a gap spot
                row_index += 1
            else:
                start_vec[mat_index][row_index] = 1
                current_block_size: int = int(size_generator.rvs())

                # guard against leaving a single element (instead expand current_block_size)
                if self.dim - (current_block_size + row_index) < block_properties.len_min:
                    current_block_size = self.dim - row_index

                # guard against overshooting the matrix size
                if current_block_size + row_index > self.dim:
                    current_block_size = self.dim - row_index
                    if current_block_size < block_properties.len_max:
                        raise ValueError("Clamped block size is too small")

                for j in range(current_block_size):
                    a = j + row_index

                    # ================================================================
                    # enable the code to force all diagonal values to be non-zero if requested
                    if self.force_nonzero_diag:
                        # allways set diagonal to value
                        value: float = np.random.uniform(value_properties.val_min, value_properties.val_max)
                        if value == 0.0:
                            value = 1.0  # this changes interval [0, 1) to (0, 1]
                        self.matrices[mat_index][a][a] = np.float32(value)
                    else:
                        # set diagonal to value
                        if np.random.random() < block_density:
                            value: float = np.random.uniform(value_properties.val_min, value_properties.val_max)
                            self.matrices[mat_index][a][a] = np.float32(value)
                    # ================================================================

                    for i in range(j):
                        b = i + row_index
                        if a == b:
                            raise ValueError("Diagonal overwrite!")
                        if np.random.random() < block_density:
                            value: float = np.random.uniform(value_properties.val_min, value_properties.val_max)
                            self.matrices[mat_index][a][b] = np.float32(value)
                            self.matrices[mat_index][b][a] = np.float32(value)
                row_index += current_block_size

        return block_density

    def __generate_value_space(self, density: float, vp: ValueProperties):
        # initialize data-matrix (values)
        data: np.ndarray = np.zeros((self.dim, self.dim), dtype=np.float32)
        # initialize truth-matrix (selector)
        sel: np.ndarray = np.zeros((self.dim, self.dim), dtype=bool)
        # populate lower triangular matrix selector
        sel[np.tril_indices(self.dim)] = np.random.uniform(size=((self.dim * (self.dim + 1)) // 2)) < density
        # add values to matrix's lower triangular based on selector
        data[sel] = np.random.uniform(vp.val_min, vp.val_max, size=sel.sum())
        # copy lower triangular back onto upper triangular via transpose
        data[np.triu_indices(self.dim)] = data.T[np.triu_indices(self.dim)]

        return data

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

    def get_list_of_block_sizes(self, block_index_array: np.ndarray) -> np.ndarray:
        # based on https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
        block_index_flat_array: np.ndarray = block_index_array.reshape(self.len * self.dim)
        # Create indicator array that is 1 where 'm' is 0 and 1 elsewhere. Pad ends with 0 for n-th diff step.
        is_zero: np.ndarray = np.concatenate(([0], np.equal(block_index_flat_array, 0).view(np.int8), [0]))
        # Calculate the n-th discrete difference (m[i+1] - m[i]) and gather in array.
        abs_nth_discrete_diff: np.ndarray = np.abs(np.diff(is_zero))
        # Get the start and end_plus_one of each series of zeroes in matrices. These are one off as '1' is block start.
        block_start_and_end = np.where(abs_nth_discrete_diff == 1)[0].reshape(-1, 2)
        block_lengths: np.ndarray = block_start_and_end[:, 1] - block_start_and_end[:, 0] + 1
        # Count the numbers to get the total entries that are blocks of size greater one.
        block_total: int = block_lengths.sum()
        # Count the amount of -1 entries which tells us the number of gap entries.
        gap_total: int = np.equal(block_index_flat_array, -1).sum()
        # Subtract blocks and gaps to get the number of blocks of size equal to one.
        one_blocks_total: int = self.len * self.dim - block_total - gap_total
        # Initialize ones array which can be empty if no one-blocks were created (e.g. block size min > 1).
        ones_lengths: np.ndarray = np.ones(one_blocks_total, dtype=block_lengths.dtype)

        return np.concatenate((block_lengths, ones_lengths))
            

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
        data_fig = pyplot.figure(num=selected_index, figsize=(14, 5))
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
