import numpy as np


def peg_algorithm(N, M, d_v, d_c, max_iter=1000):
    H = np.zeros((M, N), dtype=int)
    row_weight = np.zeros(M, dtype=int)
    col_weight = np.zeros(N, dtype=int)
    num_iter = 0

    while np.any(row_weight < d_c) and np.any(col_weight < d_v):
        if num_iter >= max_iter:
            print("Maximum iterations reached. Unable to generate matrix.")
            return None

        v = np.random.choice(np.where(col_weight < d_v)[0])
        remaining_checks = np.where(row_weight < d_c)[0]

        for _ in range(d_v - col_weight[v]):
            if len(remaining_checks) == 0:
                print("Error: Unable to find enough check nodes to connect.")
                return None

            c = np.random.choice(remaining_checks)
            H[c, v] = 1
            row_weight[c] += 1
            col_weight[v] += 1

            if row_weight[c] == d_c:
                remaining_checks = np.delete(
                    remaining_checks, np.where(remaining_checks == c)[0]
                )

        num_iter += 1

    return H


def check_node_update(llr, H):
    M, N = H.shape
    check_node_values = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            connected_bits = np.nonzero(H[i])[0]
            product = np.prod(np.tanh(0.5 * llr[connected_bits]))
            if product == 1:
                check_node_values[i, j] = np.inf
            elif product == -1:
                check_node_values[i, j] = -np.inf
            else:
                check_node_values[i, j] = -2 * np.arctanh(product)

    return check_node_values


def bit_node_update(check_node_values, H, y, sigma_squared):
    M, N = H.shape
    bit_node_values = np.zeros(N)

    for i in range(N):
        connected_checks = np.nonzero(H[:, i])[0]
        bit_node_values[i] = y[i] / sigma_squared + np.sum(
            check_node_values[connected_checks, i]
        )

    return bit_node_values


def ldpc_decode(y, H, max_iter=50, sigma_squared=0.01):
    N = len(y)
    llr = 2 * y / sigma_squared

    for _ in range(max_iter):
        check_node_values = check_node_update(llr, H)
        bit_node_values = bit_node_update(check_node_values, H, y, sigma_squared)

        llr = 2 * (
            y / sigma_squared + np.sum(check_node_values, axis=0) - check_node_values
        )

    decoded_bits = np.where(llr < 0, 1, 0)

    return decoded_bits


def ldpc_correct(decoded_bits, H, max_iter=50):
    N = len(decoded_bits)
    for _ in range(max_iter):
        decoded_bits_new = ldpc_decode(decoded_bits, H)
        if np.array_equal(decoded_bits, decoded_bits_new):
            break
        decoded_bits = decoded_bits_new
    return decoded_bits


N = 10
M = 10  # Adjust M to be equal to or greater than N
d_v = 3
d_c = 4
sigma_squared = 0.01

H = peg_algorithm(N, M, d_v, d_c)

if H is not None:
    # Corrected message: first half zeros, second half ones
    message_bits = np.concatenate(
        (np.zeros(N // 2, dtype=int), np.ones(N // 2, dtype=int))
    )

    encoded_bits = np.mod(np.dot(H, message_bits), 2)
    received_bits = encoded_bits + np.random.normal(0, np.sqrt(sigma_squared), N)
    decoded_bits = ldpc_decode(received_bits, H)

    print("Original Message:", message_bits)
    print("Received Message:", received_bits)
    print("Decoded Message (Before Correction):", decoded_bits)

    # Correct the decoded message
    corrected_bits = ldpc_correct(decoded_bits, H)

    print("Decoded Message (After Correction):", corrected_bits)
    print("Error Rate:", np.sum(np.abs(message_bits - corrected_bits)) / N)
else:
    print("Error: Unable to generate matrix.")
