# encoding: utf-8
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class MatrixData:
    __METADATA_FIELDS: int = 9  # change this when changing the MetaData class

    class MetaData:
        """A data class containing the randomized values that were assigned to the corresponding matrix with the same
        index.

        **Note:** To keep variable names reasonably short the following abbreviations are used:
            | ``bgr_...`` for background
            | ``blk_...`` for blocks
            | also note that ``tdata`` here refers to the `true` data values
        """
        bgr_noise_den: float = 0
        bgr_noise_min: float = 0
        bgr_noise_max: float = 0
        blk_noise_den: float = 0
        blk_noise_min: float = 0
        blk_noise_max: float = 0
        blk_tdata_den: float = 0
        blk_tdata_min: float = 0
        blk_tdata_max: float = 0

        def __init__(self):
            pass

    # provided input
    sample_size: int
    band_radius: int
    dimension: int
    force_invertible: bool
    # background parameters
    bgr_noise_den_min: float
    bgr_noise_den_max: float
    bgr_noise_val_min: np.float32
    bgr_noise_val_max: np.float32
    # noise block parameters
    blk_noise_den_min: float
    blk_noise_den_max: float
    blk_noise_val_min: np.float32
    blk_noise_val_max: np.float32
    # true data block parameters
    blk_tdata_den_min: float
    blk_tdata_den_max: float
    blk_tdata_val_min: np.float32
    blk_tdata_val_max: np.float32

    # debug data
    debug: bool = False

    # generated output
    matrices: np.ndarray = None
    bands: np.ndarray = None
    block_data_start_labels: np.ndarray = None
    block_noise_start_labels: np.ndarray = None
    metadata: list[MetaData] = None

    def __init__(
            self,
            dimension: int,
            band_radius: int,
            sample_size: int,
            background_noise_density_range: (float, float),
            background_noise_value_range: (np.float32, np.float32),
            block_noise_density_range: (float, float),
            block_noise_value_range: (np.float32, np.float32),
            block_data_density_range: (float, float),
            block_data_value_range: (np.float32, np.float32),
            force_invertible: bool = False,
            print_debug: bool = False,
    ):
        self.dimension = dimension
        self.sample_size = sample_size
        self.band_radius = band_radius
        self.bgr_noise_den_min, self.bgr_noise_den_max = background_noise_density_range
        self.bgr_noise_val_min, self.bgr_noise_val_max = background_noise_value_range
        self.blk_noise_den_min, self.blk_noise_den_max = block_noise_density_range
        self.blk_noise_val_min, self.blk_noise_val_max = block_noise_value_range
        self.blk_tdata_den_min, self.blk_tdata_den_max = block_data_density_range
        self.blk_tdata_val_min, self.blk_tdata_val_max = block_data_value_range

        self.force_invertible = force_invertible
        self.debug = print_debug

        self.__init_data_size()
        self.__generate_matrices()

    def __init_data_size(self) -> None:
        n: int = self.sample_size
        dim: int = self.dimension
        r: int = self.band_radius

        self.matrices = np.zeros(shape=(n, dim, dim), dtype=np.float32)
        self.block_data_start_labels = np.zeros(shape=(n, dim), dtype=bool)
        self.block_noise_start_labels = np.zeros(shape=(n, dim), dtype=bool)
        self.bands = np.zeros(shape=(n, dim, r), dtype=np.float32)

        self.metadata = []
        for i in range(n):
            self.metadata.append(MatrixData.MetaData())

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
        self.__add_background_noise()

    def __add_background_noise(self) -> None:
        # create some random noise values (some might be overridden later)
        for n in range(self.sample_size):
            noise_density = np.random.uniform(self.bgr_noise_den_min, self.bgr_noise_den_max)
            self.metadata[n].bgr_noise_den = noise_density  # store generated noise density in metadata field
            for j in range(self.dimension):
                for i in range(j):
                    if np.random.random() < noise_density:
                        value: float = np.random.uniform(self.bgr_noise_val_min, self.bgr_noise_val_max)
                        self.matrices[n][j][i] = np.float32(value)
                        self.matrices[n][i][j] = np.float32(value)


if __name__ == "__main__":
    test_matrices = MatrixData(
        dimension=64, band_radius=10, sample_size=1000,
        background_noise_density_range=(0.3, 0.5), background_noise_value_range=(0.0, 0.5),
        block_noise_density_range=(0.3, 0.5), block_noise_value_range=(0.3, 1.0),
        block_data_density_range=(0.5, 0.7), block_data_value_range=(0.3, 1.0),
        print_debug=True
    )

    data_fig = plt.figure(num=1, figsize=(6, 5))
    sns.heatmap(
        test_matrices.matrices[0],
        cmap='rocket',
        cbar_kws={'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
        xticklabels=False,
        yticklabels=False,
        square=True,
        vmin=0,
        vmax=1
    )
    data_fig.show()
