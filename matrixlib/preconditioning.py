import numpy as np
import scipy
from scipy.sparse.linalg import gmres, LinearOperator


# def ensure_nonsingularity(A: np.ndarray, band_width: int = 10) -> np.ndarray:
#     """
#     Modifies the input matrices to ensure non-singularity by replacing all nonzero entries
#     with values in the range (-1, 0) and setting all diagonal band values to 1.0.
#
#     Args:
#         A: NumPy array of shape (n, m, m) representing n square matrices of size m x m.
#         band_width: Width of the diagonal band to set to 1.0 (default is 1, which is just the main diagonal)
#
#     Returns:
#         Modified NumPy array.
#     """
#     A_mod = np.copy(A)
#     n, m, _ = A.shape
#
#     for k in range(n):
#         # Replace nonzero off-diagonal elements with uniform random values in (-1, 0)
#         mask = (A_mod[k] != 0) & ~np.eye(m, dtype=bool)
#         A_mod[k][mask] = np.random.uniform(-1, 0, size=np.sum(mask))
#
#         # Set diagonal band to 1.0
#         for i in range(m):
#             for j in range(max(0, i-band_width+1), min(m, i+band_width)):
#                 A_mod[k, i, j] = 1.0
#
#     return A_mod

def block_jacobi_preconditioner_from_predictions(input_matrix: np.ndarray,
                                                 prediction_indicator_array: np.ndarray) -> np.ndarray:
    """
        Compute a block Jacobi preconditioner based on predicted block structure.

        This function creates a preconditioner matrix by inverting blocks of the input matrix
        and applying min-max normalization. The block structure is determined by the
        prediction_indicator_array.

        Parameters:
        input_matrix (np.ndarray): The input matrix to be preconditioned. Shape: (n, m, m)
        prediction_indicator_array (np.ndarray): Array indicating the start of each block.
                                                 Shape: (n, m)

        Returns:
        np.ndarray: The computed preconditioner matrix. Shape: (n, m, m)

        Note:
        - The function inverts each block of the input matrix.
        - After inversion, min-max normalization is applied and values are inverted.
        - The diagonal elements of the final preconditioner are set to 1.0.
        """
    n, m, _ = input_matrix.shape
    prec = np.zeros_like(input_matrix)

    for k in range(n):

        # Convert block start flags on array len=dim to list of indices of block starts
        block_starts = np.where(prediction_indicator_array[k] == 1)[0]
        # Add dim to end of this array so that the last block ends at the end of the matrix
        block_starts = np.append(block_starts, m)

        for i in range(len(block_starts) - 1):
            start = block_starts[i]
            end = block_starts[i + 1]
            block = input_matrix[k, start:end, start:end]
            prec[k, start:end, start:end] = scipy.linalg.inv(block)  # Invert each block

        # Normalise nonzero elements to range (-1, 0)
        # val_min, val_max = prec[k].min(), prec[k].max()
        # prec[k] = -1 + (prec[k] - val_min) / (val_max - val_min)
        # prec[k][np.diag_indices(m)] = 1.0

    return prec


def prepare_matrix(A: np.ndarray) -> np.ndarray:
    """
    Modifies the input matrix to ensure non-singularity by replacing all nonzero entries with values in the range (-1, 0) and setting all diagonal values to 1.0.

    Args:
    :param A: NumPy array of shape (n, m, m) representing n square matrices of size m x m.
    :return: NumPy array of shape (n, m, m) with modified values.
    """
    A_prep = A.copy()

    # Identify nonzero elements using boolean mask
    nonzero_mask = A_prep != 0

    # Normalise nonzero elements to range (-1, 0)
    nonzero_vals = A_prep[nonzero_mask]
    min_val, max_val = nonzero_vals.min(), nonzero_vals.max()
    A_prep[nonzero_mask] = -1 + (nonzero_vals - min_val) / (max_val - min_val)

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

    for k in range(n):
        iteration_count = [0]
        residuals = []

        def callback(rk, xk=None, sk=None):
            iteration_count[0] += 1
            residuals.append(rk)

        if M is not None:
            M_op = LinearOperator(matvec=lambda x: M[k] @ x, shape=(m, m))  # Apply preconditioner by multiplication
            x, info = gmres(A[k], b[k], x0=np.zeros_like(b[k]), M=M_op, rtol=rtol, callback=callback,
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

        return x_solutions, info_array, iteration_counts, all_residuals
