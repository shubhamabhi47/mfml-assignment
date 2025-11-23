import numpy as np

def cholesky_decomposition(A, fix_symmetry=False):
    """
    Computes the Cholesky decomposition of a symmetric, positive-definite matrix A.
    Returns lower triangular matrix L such that A = L @ L.T.
    
    Args:
        A (np.ndarray): Symmetric positive-definite matrix (n x n)
        fix_symmetry (bool): If True, symmetrizes A by (A + A.T)/2
    
    Returns:
        np.ndarray: Lower triangular matrix L
    """
    
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    # Optional symmetry correction
    if not np.allclose(A, A.T):
        if fix_symmetry:
            A = 0.5 * (A + A.T)
        else:
            raise ValueError("Matrix must be symmetric for Cholesky decomposition.")

    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):

            if i == j:
                # Diagonal
                sum_sq = np.sum(L[i, :j] ** 2)
                term = A[i, i] - sum_sq

                if term <= 0:
                    raise ValueError(
                        f"Matrix is not positive definite at diagonal element ({i},{i})."
                    )
                L[i, i] = np.sqrt(term)

            else:
                # Off-diagonal
                sum_prod = np.sum(L[i, :j] * L[j, :j])
                if np.isclose(L[j, j], 0.0):
                    raise ValueError(
                        f"Zero diagonal encountered at L[{j},{j}], matrix not positive definite."
                    )

                L[i, j] = (A[i, j] - sum_prod) / L[j, j]

    return L


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    A = np.array([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
    ], dtype=float)

    print("--- Input Matrix A ---")
    print(A)

    L = cholesky_decomposition(A)

    print("\n--- Cholesky Lower-Triangular Matrix L ---")
    print(L)

    print("\n--- L Transpose (Upper Triangular) ---")
    print(L.T)

    print("\n--- Verification (L @ L.T) ---")
    print(L @ L.T)

    if np.allclose(A, L @ L.T):
        print("\n✓ Verification Successful: A ≈ L @ L.T")
    else:
        print("\n✗ Verification Failed")
