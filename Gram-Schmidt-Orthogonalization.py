import numpy as np

def gram_schmidt_qr(A):
    """
    Computes the QR decomposition of matrix A using the Gram-Schmidt process.
    (A = Q * R, where Q is orthonormal and R is upper triangular)

    Args:
        A (np.ndarray): The input matrix (m x n).

    Returns:
        tuple: (Q, R), where Q is the orthonormal matrix and R is the upper
               triangular matrix.
    """
    # Use float type for accurate calculations
    A = A.astype(float)
    m, n = A.shape
    
    # Initialize Q (orthonormal basis) and R (upper triangular) matrices
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    # Iterate over the columns of A
    for j in range(n):
        v = A[:, j] # Start with the j-th column of A (a_j)
        
        # Project a_j onto the subspace spanned by the previously found orthogonal vectors (q_0, ..., q_{j-1})
        for i in range(j):
            # R[i, j] is the coefficient for projection: r_ij = <a_j, q_i>
            # Since Q is orthonormal, R[i, j] = Q[:, i].T @ v (projection length)
            R[i, j] = np.dot(Q[:, i], v)
            
            # Subtract the projection from v
            v = v - R[i, j] * Q[:, i]
            
        # The remaining vector v is orthogonal to all q_i (i < j).
        
        # R[j, j] is the norm of the new orthogonal vector v
        R[j, j] = np.linalg.norm(v)
        
        # Check for linear dependence (norm being zero or close to zero)
        if np.isclose(R[j, j], 0.0):
            print(f"Warning: Column {j} is linearly dependent on previous columns. Q[:, {j}] will be zero.")
            # If linearly dependent, Q[:, j] remains the zero vector from initialization.
        else:
            # Normalize v to get the orthonormal vector q_j
            Q[:, j] = v / R[j, j]

    return Q, R

# --- Example Usage ---
if __name__ == '__main__':
    # Define a matrix A
    A_input = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ])

    print("--- Input Matrix A ---")
    print(A_input)

    Q_result, R_result = gram_schmidt_qr(A_input)

    print("\n--- Resulting Orthonormal Matrix Q ---")
    print(Q_result)

    print("\n--- Resulting Upper Triangular Matrix R ---")
    print(R_result)

    # Verification: Q @ R should be close to A
    A_reconstructed = np.dot(Q_result, R_result)
    print("\n--- Verification (Q @ R) ---")
    print(A_reconstructed)
    
    if np.allclose(A_input, A_reconstructed):
        print("\nVerification successful: A is close to Q @ R")
    else:
        print("\nVerification failed.")

    # Verification of Q's orthonormality: Q.T @ Q should be the identity matrix
    QTQ = np.dot(Q_result.T, Q_result)
    print("\n--- Verification of Q's Orthonormality (Q.T @ Q) ---")
    print(QTQ)
    # The result should be very close to an Identity matrix (diag 1.0, off-diag 0.0)