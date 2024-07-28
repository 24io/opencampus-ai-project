# encoding: utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.linalg import inv, pinv, LinAlgError
from scipy.sparse.linalg import gmres
from matrixkit import util


def generate_block_jacobi_preconditioner(
        sparse_matrices: np.ndarray,
        block_start_indicators: np.ndarray,
        apply_inverse_minmax_norm: bool = True
) -> np.ndarray:
    """Compute a block Jacobi preconditioner from a matrix and its block start indicator.

    This function creates a preconditioner matrix by inverting blocks of the input matrix and applying min-max
    normalization. The block structure is determined by the ``block_start_indicator`` array.

    :param sparse_matrices: An array of `symmetrical` and `sparse` input matrices on which to operate.
    :param block_start_indicators: A block start indicator of the input matrices where ones denote starts of blocks and
        zeros denote ends of blocks. Each matrix must start with a block.
    :param apply_inverse_minmax_norm: A flag to indicate if an inverse minmax normalization should be applied.

    :returns np.ndarray: The array of computed preconditioner matrices.

    Note:
    - The function inverts each block of the input matrix.
    - After inversion, min-max normalization is applied and values are inverted.
    - The diagonal elements of the final preconditioner are set to 1.0.
    """
    n: int  # number of the matrices
    m: int  # dimension of the symmetrical matrices
    n, m, _ = sparse_matrices.shape

    precon: np.ndarray = np.zeros_like(sparse_matrices)
    for k in range(n):
        # Convert block start indicator arrays to arrays of indices indicating block starts. As block starts also mirror
        # as block ends (exclusive), an entry of the dimension is added to the end of this array.
        block_starts: np.ndarray = np.append(np.where(block_start_indicators[k] == 1)[0], m)

        for i in range(len(block_starts) - 1):
            start = block_starts[i]
            end = block_starts[i + 1]
            block = sparse_matrices[k, start:end, start:end]

            try:
                precon[k, start:end, start:end] = inv(block)  # Invert each block
            except LinAlgError as e:
                if e.args[0] == "singular matrix":
                    print(f"Block is singular, using pseudo-inverse for block {k:4d} at indices {start}:{end}")
                    precon[k, start:end, start:end] = pinv(block)
                else:
                    raise e

    if apply_inverse_minmax_norm:
        # Map values of the preconditioner 'A' with f: [min, max] -> [-1, 0], f(A) = -1 * minmax(A).
        precon = util.apply_minmax_norm(precon, offset=-1.0)
        for k in range(n):
            # Set all diagonal values to one. In summary this and the previous operation should prevent singularities.
            precon[k][np.diag_indices(m)] = 1.0

    return precon


def prepare_matrix(input_matrix: np.ndarray, mapping_type: str = "flip") -> np.ndarray:
    """ Modifies the input matrix to ensure non-singularity by replacing all nonzero entries with values in the
    interval (-1, 0) and setting all diagonal values to 1.0.

    :param input_matrix: A ``np.ndarray`` of shape (``n``, ``m``, ``m``) representing n square matrices.
    :param mapping_type: One of ['flip', 'flip_norm', 'shift_norm'] to control the type of preparation.
        'flip' just applies a factor of -1. Intended for matrices with values in [0, 1]
        'flip_norm' first applies the minmax-norm and then a factor of -1
        'shift_norm' first applies the minmax-norm and then an offset of -1
    :return: A ``np.ndarray`` of shape (``n``, ``m``, ``m``) with modified values.
    """
    number_of_matrices: int
    if not (2 <= len(input_matrix.shape) <= 3):
        raise ValueError(f"Provided 'input_matrix' must be of shape (n, m, m) or (m, m). Got {input_matrix.shape}")

    number_of_matrices = 1 if len(input_matrix.shape) == 2 else input_matrix.shape[0]

    result_matrix: np.ndarray = input_matrix.copy()
    match mapping_type:
        case "minmax": util.apply_minmax_norm(result_matrix)
        case "flip":        result_matrix *= -1.0
        case "flip_norm":   util.apply_minmax_norm(result_matrix, factor=-1.0)
        case "shift":       result_matrix += -1.0
        case "shift_norm":  util.apply_minmax_norm(result_matrix, offset=-1.0)
        case _: raise ValueError(f"Invalid mapping_type '{mapping_type}'")

    if len(input_matrix.shape) == 2:
        np.fill_diagonal(result_matrix, 1.0)
    else:
        for i in range(number_of_matrices):
            np.fill_diagonal(result_matrix[i], 1.0)

    return result_matrix


def solve_with_gmres_monitored(
        matrix: np.ndarray,
        b_vector: np.ndarray,
        preconditioner: np.ndarray = None,
        relative_tolerance: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Solve a system of linear equations using GMRES with optional preconditioning and monitoring.

    This function solves ``Ax = b`` for given right-hand side ``b`` and matrix ``A`` using the Generalized Minimal
    Residual method (GMRES).
    It supports optional preconditioning and monitors the number of convergence steps.

    :param matrix: A ``np.ndarray`` of shape (``n``, ``m``, ``m``) representing several coefficient matrices.
    :param b_vector: A ``np.ndarray`` of shape (``n``, ``m``) representing the corresponding right-hand side vectors.
    :param preconditioner: A ``np.ndarray`` of shape (``n``, ``m``, ``m``) representing the preconditioner matrices.
    :param relative_tolerance: The relative tolerance for convergence. Default is 1e-3.
    :return: a tuple of the solution vector and some metadata generated during the inversion steps.

    The returned tuple consists of the following elements:
        - ``x_solutions``: ``np.ndarray`` of shape (``n``, ``m``) representing the solution vectors ``x``.
        - ``info_array``: ``np.ndarray`` of shape (``n``) holding success information of the solver for each system.
        - ``iteration_counts``: ``np.ndarray`` of shape (``n``) holding the number of iterations for each system.
        - ``all_residuals``: ``list`` of residual norms for each system.

    Note:
        - The function solves n separate linear systems, one for each slice of A and b.
        - If a preconditioner is provided, it is applied as a `left` preconditioner.
        - The function monitors and returns the `residual norms` at each iteration.
    """
    n, m, _ = matrix.shape
    x_solutions = np.zeros_like(b_vector)
    info_array = np.zeros(n, dtype=int)
    iteration_counts = np.zeros(n, dtype=int)
    all_residuals: list[list[float]] = []

    def callback(residual_norm: float):
        iteration_count[0] += 1
        residuals.append(residual_norm)

    for k in range(n):
        iteration_count = [0]
        residuals: list[float] = []

        run_matrix: np.ndarray = matrix[k]
        run_b_vector: np.ndarray = b_vector[k]
        if preconditioner is not None:
            run_matrix = preconditioner[k] @ matrix[k]
            run_b_vector = preconditioner[k] @ b_vector[k]

        x, info = gmres(
            A=run_matrix,
            b=run_b_vector,
            x0=np.zeros_like(run_b_vector),
            rtol=relative_tolerance,
            callback=callback,
            callback_type='pr_norm'
        )
        x_solutions[k] = x
        info_array[k] = info
        iteration_counts[k] = iteration_count[0]
        all_residuals.append(residuals)

    return x_solutions, info_array, iteration_counts, all_residuals


def run_gmres_experiments(input_matrices_list, preconditioners_list, b):
    results = []

    for input_name, input_matrices in input_matrices_list:
        for prec_name, preconditioner_source in preconditioners_list:
            print(f'Solving {input_name} with {prec_name}')
            x, info, iters, residuals = solve_with_gmres_monitored(input_matrices, b, preconditioner_source)

            converged = np.sum(info == 0)
            total_systems = len(info)
            percent_converged = (converged / total_systems) * 100

            results.append({
                'Matrix Type': input_name,
                'Preconditioner Type': prec_name,
                'Converged': converged,
                'Total Systems': total_systems,
                'Percent Converged': percent_converged,
                'Mean Iterations': np.mean(iters),
                'Median Iterations': np.median(iters),
                'Max Iterations': np.max(iters),
                'Min Iterations': np.min(iters)
            })

    return pd.DataFrame(results)


def print_results_table(df):
    print(df.to_string(index=False))


def plot_results(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Matrix Type', y='Mean Iterations', hue='Preconditioner Type', data=df)
    plt.title('Mean Iterations by Matrix Type and Preconditioner')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Matrix Type', y='Percent Converged', hue='Preconditioner Type', data=df)
    plt.title('Percent Converged by Matrix Type and Preconditioner')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # only import this when running manual tests
    import core
    from numpy.linalg import cond

    MATRIX_DIM = 64
    NUMBER_OF_MATRICES = 100
    DIAGONAL_BAND_RADIUS = 10

    bgr_noise_value_props = core.ValueProperties(density_range=(0.3, 0.5), value_range=(0.0, 0.5))
    noise_blk_value_props = core.ValueProperties(density_range=(0.3, 0.5), value_range=(0.3, 1.0))
    noise_blk_block_props = core.BlockProperties(size_range=(3, 32), size_average=10, size_std_dev=0.66, gap_chance=0.5)
    tdata_blk_value_props = core.ValueProperties(density_range=(0.5, 0.7), value_range=(0.3, 1.0))
    tdata_blk_block_props = core.BlockProperties(size_range=(2, 32), size_average=10, size_std_dev=0.66, gap_chance=0.0)

    test_data = core.MatrixData(
        dimension=MATRIX_DIM,
        band_radius=DIAGONAL_BAND_RADIUS,
        sample_size=NUMBER_OF_MATRICES,
        background_noise_value_properties=bgr_noise_value_props,
        block_noise_value_properties=noise_blk_value_props,
        block_noise_block_properties=noise_blk_block_props,
        block_data_value_properties=tdata_blk_value_props,
        block_data_block_properties=tdata_blk_block_props,
        seed=42,
        determinant_cutoff=0.01,
        print_debug=False
    )

    # prepare test_data with negative minmax norm and diagonal ones.
    matrices = test_data.matrices
    prepared_test_data = prepare_matrix(input_matrix=test_data.matrices, mapping_type="shift")
    precon_data = generate_block_jacobi_preconditioner(
        test_data.matrices, test_data.tdata_blk_starts, apply_inverse_minmax_norm=True)

    print("-" * 80)
    condition_numbers: np.ndarray = np.asarray([cond(precon_data[i]) for i in range(NUMBER_OF_MATRICES)])
    print(f"preconditioner condition numbers: {condition_numbers[0:5]}")

    # define all-ones solution vector
    solution_vector: np.ndarray = np.ones((NUMBER_OF_MATRICES, MATRIX_DIM))

    print("-" * 80)
    # run GMRES with no preconditioner
    print("Starting GMRES run with no preconditioner")
    x_no_precon, info_no_precon, iters_no_precon, residuals_no_precon = solve_with_gmres_monitored(
        matrix=matrices, b_vector=solution_vector, preconditioner=None, relative_tolerance=1e-3)
    print("-" * 80)
    # un GMRES with block Jacobi preconditioner
    print("Starting GMRES run with preconditioner")
    x_precon, info_precon, iters_precon, residuals_precon = solve_with_gmres_monitored(
        matrix=matrices, b_vector=solution_vector, preconditioner=precon_data, relative_tolerance=1e-3)

    print("Determinants of singular (non-solvable) matrices")
    iterations: np.ndarray = iters_precon
    singular_indices = np.where(iterations == 12800)[0]
    selected_determinants = [test_data.metadata[i].det for i in singular_indices]
    print(selected_determinants)

    print("All done.")
