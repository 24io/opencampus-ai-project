import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc
from scipy.sparse.csgraph import reverse_cuthill_mckee


def supervariable_blocking(input_matrices: np.ndarray, max_block_size: int = 32):
    """
        Performs supervariable blocking on a set of input matrices using Chow et al. (2018)'s proposed approach
        (https://icl.utk.edu/files/publications/2018/icl-utk-1067-2018.pdf p.5)

        This function takes a set of input matrices, performs Reverse Cuthill-McKee (RCM) reordering,
        extracts the sparsity patterns of the reordered matrices, identifies supervariables by comparing
        sparsity patterns of adjacent columns, and amalgamates these supervariables into blocks with a
        given maximum block size.

        ----------
        Parameters:
            :param input_matrices: NumPy array of shape (n, m, m) representing n square matrices of size m x m.
            :param max_block_size: The maximum size of each block. Default is 32.

        Returns:
            :return block_starts_array: NumPy array of shape (n, m) where each element is either 0 or 1.
            A value of 1 at position (i, j) indicates the start of a block in the i-th matrix at column j.
        -----------
        Notes:
        ------
        - This function assumes that the input matrices are sparse and can be converted to CSC (Compressed Sparse Column) format.
        - The RCM reordering helps in reducing the bandwidth of the sparse matrices, which is useful in identifying supervariables.
        - Supervariables are identified based on identical sparsity patterns in adjacent columns of the reordered matrix.
        - Blocks are formed by merging supervariables together, ensuring that the size of each block does not exceed the specified maximum block size.
        - The final block starts are recorded in the `block_starts_array`, indicating the starting positions of the blocks in the original matrix ordering.
        - Output is compatible with the `block_jacobi_preconditioner_from_predictions` function for creating block Jacobi preconditioners.
        """

    n, m, _ = input_matrices.shape
    all_supervariables = []
    block_starts_array = np.zeros((n, m), dtype=int)

    for i in range(n):
        matrix = input_matrices[i]
        if not isspmatrix_csc(matrix):
            matrix = csc_matrix(matrix)

        # Perform RCM reordering
        permutation = reverse_cuthill_mckee(matrix, symmetric_mode=True)
        reordered_matrix = matrix[permutation][:, permutation]

        # Extract the sparsity pattern of the reordered matrix
        sparsity_patterns = []
        for col in range(m):
            pattern = set(reordered_matrix.indices[reordered_matrix.indptr[col]:reordered_matrix.indptr[col + 1]])
            sparsity_patterns.append(pattern)

        # Identify supervariables by comparing sparsity patterns of adjacent columns
        supervariables = []
        current_supervariable = [0]
        for j in range(1, m):
            if sparsity_patterns[j] == sparsity_patterns[j - 1]:
                current_supervariable.append(j)
            else:
                supervariables.append(current_supervariable)
                current_supervariable = [j]
        if current_supervariable:
            supervariables.append(current_supervariable)

        # Amalgamate (merge) supervariables into blocks with a given maximum block size
        blocks = []
        current_block = []
        current_block_size = 0
        for supervariable in supervariables:
            if current_block_size + len(supervariable) <= max_block_size:
                current_block.extend(supervariable)
                current_block_size += len(supervariable)
            else:
                blocks.append(current_block)
                current_block = supervariable
                current_block_size = len(supervariable)
        if current_block:
            blocks.append(current_block)

        # Convert blocks back to original matrix ordering
        original_blocks = [[permutation[i] for i in block] for block in blocks]
        all_supervariables.append(original_blocks)

        # Record block start indices
        block_starts = [block[0] for block in original_blocks]
        block_starts_array[i, block_starts] = 1

    return block_starts_array
