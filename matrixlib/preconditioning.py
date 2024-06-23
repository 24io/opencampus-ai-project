# encoding: utf-8
import numpy as np

from scipy.linalg import inv, pinv, LinAlgError
from scipy.sparse.linalg import gmres

from matrixlib import util


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
    if 2 <= len(input_matrix.shape) <= 3:
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


def solve_with_gmres_monitored(A: np.ndarray, b: np.ndarray, M: np.ndarray = None, rtol: float = 1e-3) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list]:
    """
        Solve a system of linear equations using GMRES with optional preconditioning and monitoring.

        This function solves Ax = b for multiple right-hand sides using the Generalized Minimal Residual method (GMRES).
        It supports optional preconditioning and monitors the convergence process.

        Parameters:
        A (np.ndarray): Coefficient matrix. Shape: (n, m, m)
        b (np.ndarray): Right-hand side vector. Shape: (n, m)
        M (np.ndarray, optional): Preconditioner matrix. Shape: (n, m, m). Default is None.
        maxiter (int, optional): Maximum number of iterations. Default is 1000.
        rtol (float, optional): Relative tolerance for convergence. Default is 1e-3.

        Returns:
        tuple:
            - x_solutions (np.ndarray): Solution vectors. Shape: (n, m)
            - info_array (np.ndarray): Information about the success of the solver for each system. Shape: (n,)
            - iteration_counts (np.ndarray): Number of iterations for each system. Shape: (n,)
            - all_residuals (list): List of residual norms for each system.

        Note:
        - The function solves n separate linear systems, one for each slice of A and b.
        - If a preconditioner M is provided, it is applied as a left preconditioner.
        - The function monitors and returns the residual norms at each iteration.
        """
    n, m, _ = A.shape
    x_solutions = np.zeros_like(b)
    info_array = np.zeros(n, dtype=int)
    iteration_counts = np.zeros(n, dtype=int)
    all_residuals = []

    def callback(rk, xk=None, sk=None):
        iteration_count[0] += 1
        residuals.append(rk)

    for k in range(n):
        iteration_count = [0]
        residuals = []

        if M is not None:
            # M_op = LinearOperator(matvec=lambda x: M[k] @ x, shape=(m, m))  # Apply preconditioner by multiplication
            x, info = gmres(A[k], b[k], x0=np.zeros_like(b[k]), M=M[k], rtol=rtol, callback=callback,
                            callback_type='pr_norm')
        else:
            x, info = gmres(A[k], b[k], x0=np.zeros_like(b[k]), rtol=rtol, callback=callback,
                            callback_type='pr_norm')

        x_solutions[k] = x
        info_array[k] = info
        iteration_counts[k] = iteration_count[0]
        all_residuals.append(residuals)
        if k % (n // 10) == 0:
            print("*", end="")

    print("")  # end the asterix-chain

    # Print summary statistics
    print("-" * 80)
    print(f"{'With preconditioner:' if M is not None else 'Without preconditioner:'}")
    print(f"  Converged: {np.sum(info_array == 0)} out of {len(info_array)}")
    print(f"  Average iterations: {np.mean(iteration_counts):.2f}")
    print(f"  iterations: {iteration_counts}")
    print("-" * 80)

    return x_solutions, info_array, iteration_counts, all_residuals
