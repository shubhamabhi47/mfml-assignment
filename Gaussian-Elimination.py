import numpy as np

def gaussian_elimination(A, b):
    """
    Solves a system of linear equations Ax = b using Gaussian Elimination.

    Args:
        A (np.ndarray): The coefficient matrix (n x n).
        b (np.ndarray): The constant vector (n x 1).

    Returns:
        np.ndarray: The solution vector x, or None if the matrix is singular.
    """
    if A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        print("Error: Matrix A must be square and dimensions must match vector b.")
        return None

    n = A.shape[0]
    # Create the augmented matrix [A | b]
    Ab = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))

    print("--- Starting Augmented Matrix [A | b] ---")
    print(Ab)

    # Forward Elimination
    for i in range(n):
        # Find pivot (row with largest absolute value in the current column)
        pivot_row = i + np.argmax(np.abs(Ab[i:, i]))

        # Swap rows to bring the largest absolute pivot to the current position (i)
        Ab[[i, pivot_row]] = Ab[[pivot_row, i]]

        # If the pivot is zero, the matrix is singular (or nearly singular).
        if np.isclose(Ab[i, i], 0.0):
            print(f"Matrix is singular (pivot at row {i} is zero).")
            return None

        # Normalize the pivot row (make the pivot element 1)
        Ab[i] = Ab[i] / Ab[i, i]

        # Eliminate all other entries below the pivot
        for j in range(i + 1, n):
            factor = Ab[j, i]
            Ab[j] = Ab[j] - factor * Ab[i]
            
        print(f"\nMatrix after column {i} elimination:")
        print(Ab)

    print("\n--- Matrix in Row Echelon Form ---")
    
    # Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # x[i] = (b'[i] - A'[i, i+1:] * x[i+1:]) / A'[i, i]
        # Since we normalized A'[i, i] to 1, this simplifies:
        x[i] = Ab[i, n] - np.dot(Ab[i, i+1:n], x[i+1:n])

    return x

# --- Example Usage ---
if __name__ == '__main__':
    # Define a coefficient matrix A and constant vector b for the system:
    # 2x + 1y - 1z = 8
    # -3x - 1y + 2z = -11
    # -2x + 1y + 2z = -3
    A_input = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ])
    b_input = np.array([8, -11, -3])

    solution_x = gaussian_elimination(A_input, b_input)

    if solution_x is not None:
        print("\n--- Final Solution Vector x ---")
        print(solution_x)

        # Verification: A * x should be close to b
        verification = np.dot(A_input, solution_x)
        print("\nVerification (A * x):")
        print(verification)
        print("Expected (b):")
        print(b_input)