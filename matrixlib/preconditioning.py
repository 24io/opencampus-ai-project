import numpy as np
import scipy as sp
from scipy.sparse.linalg import gmres


def create_block_jacobi_preconditioner(input_matrices: np.ndarray, block_start_indicator: np.ndarray) -> np.ndarray:
    """Compute a block Jacobi preconditioner from a matrix and its block start indicator.

    This function creates a preconditioner matrix by inverting blocks of the input matrix and applying min-max
    normalization. The block structure is determined by the ``block_start_indicator`` array.

    :param input_matrices: An array of `symmetrical` input matrices on which to operate.
    :param block_start_indicator: A block start indicator of the input matrices where ones denote starts of blocks and
        zeros denote ends of blocks. Each matrix must start with a block.

    :returns np.ndarray: The array of computed preconditioner matrices.

    Note:
    - The function inverts each block of the input matrix.
    - After inversion, min-max normalization is applied and values are inverted.
    - The diagonal elements of the final preconditioner are set to 1.0.
    """
    n: int  # number of the matrices
    m: int  # dimension of the symmetrical matrices
    n, m, _ = input_matrices.shape

    precon: np.ndarray = np.zeros_like(input_matrices)
    for k in range(n):

        # Convert block start indicator arrays to arrays of indices indicating block starts. As block starts also mirror
        # as block ends (exclusive), an entry of the dimension is added to the end of this array.
        block_starts: np.ndarray = np.append(np.where(block_start_indicator[k] == 1)[0], m)

        for i in range(len(block_starts) - 1):
            start = block_starts[i]
            end = block_starts[i + 1]
            block = input_matrices[k, start:end, start:end]
            precon[k, start:end, start:end] = sp.linalg.inv(block)  # Invert single block

        # Normalise nonzero elements to range (-1, 0)
        val_min, val_max = precon[k].min(), precon[k].max()
        precon[k] = -1 + (precon[k] - val_min) / (val_max - val_min)
        precon[k][np.diag_indices(m)] = 1.0

    return precon


def prepare_matrix(A: np.ndarray) -> np.ndarray:
    """
    Modifies the input matrix to ensure non-singularity by replacing all nonzero entries with values in the range (-1, 0) and setting all diagonal values to 1.0.

    Args:
    :param A: NumPy array of shape (n, m, m) representing n square matrices of size m x m.
    :return: NumPy array of shape (n, m, m) with modified values.
    """
    A_prep = A.copy()

    # # Identify nonzero elements using boolean mask
    # nonzero_mask = A_prep != 0
    #
    # # Normalise nonzero elements to range (-1, 0)
    # nonzero_vals = A_prep[nonzero_mask]
    # min_val, max_val = nonzero_vals.min(), nonzero_vals.max()
    # A_prep[nonzero_mask] = -1 + (nonzero_vals - min_val) / (max_val - min_val)

    # flip values from [0, 1]  to [-1, 0]
    A_prep -= 1

    # Set diagonal to 1.0
    np.fill_diagonal(A_prep, 1.0)

    return A_prep


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

    # Print summary statistics
    print(f"{'With preconditioner:' if M is not None else 'Without preconditioner:'}")
    print(f"  Converged: {np.sum(info_array == 0)} out of {len(info_array)}")
    print(f"  Average iterations: {np.mean(iteration_counts):.2f}")
    print(f"  iterations: {iteration_counts}")

    return x_solutions, info_array, iteration_counts, all_residuals
