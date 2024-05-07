import numpy as np
import itertools
import random


def generator_matrix(parity_matrix):
    m, k = parity_matrix.shape
    I = np.eye(m, dtype=int)
    G = np.concatenate((parity_matrix, I), axis=1)
    return G


def generate_ldpc_code(n, k, parity_vector):
    # Step 1: Generate Parity Matrix
    parity_matrix = np.zeros((n - k, k), dtype=int)
    for i in range(n - k):
        parity_matrix[i] = np.roll(parity_vector, i)
    print("\nParity Matrix:")
    print(parity_matrix)

    # Step 2: Generate Generator Matrix
    gen_matrix = generator_matrix(parity_matrix)
    print("\nGenerator Matrix:")
    print(gen_matrix)

    # Step 3: Generate all message code and codeword
    all_message_code = np.array(list(itertools.product([0, 1], repeat=k)))

    # Step 4: Find all codewords for respective message code
    codewords_dict = {}
    for i in range(len(all_message_code)):
        parity_values = []
        for j in range(n - k):
            parity_values.append(np.dot(all_message_code[i], parity_matrix[j].T) % 2)
        codeword = np.concatenate((all_message_code[i], parity_values))
        codewords_dict[str(all_message_code[i])] = codeword
    print("\nCodewords Dictionary:")
    for code, codeword in codewords_dict.items():
        print(f"{code} : {codeword}")

    # Step 5: Generate Parity Check Matrix
    parity_check_matrix = np.concatenate(
        (parity_matrix, np.eye(n - k, dtype=int)), axis=1
    ).T
    print("\nParity Check Matrix:")
    print(parity_check_matrix)

    return generator_matrix, parity_check_matrix, codewords_dict


def flip_bit(codeword, position):
    flipped_codeword = codeword.copy()
    flipped_codeword[position] = 1 - flipped_codeword[position]
    return flipped_codeword


def ldpc_decode(received_codeword, parity_check_matrix):
    syndrome = np.dot(received_codeword, parity_check_matrix) % 2
    return syndrome


def ldpc_correct(received_codeword, parity_check_matrix):
    syndrome = ldpc_decode(received_codeword, parity_check_matrix)
    error_position = np.where(np.all(parity_check_matrix == syndrome, axis=1))[0]
    if len(error_position) == 0:
        print("\nNo error detected.")
        return received_codeword
    else:
        print(f"\nError detected at bit position(s) {error_position}. Correcting...")
        corrected_codeword = flip_bit(received_codeword, error_position[0])
        print("Corrected codeword:", corrected_codeword)
        return corrected_codeword


n = 7  # Total number of bits
k = 4  # Number of message bits
print("length of codeword:", n)
print("Number of message bits:", k)
parity_vector = np.random.randint(0, 2, size=(k,))
# parity_vector = [1, 1, 1, 0]
print("\nParity Vector:", parity_vector)

generator_matrix, parity_check_matrix, codewords_dict = generate_ldpc_code(
    n, k, parity_vector
)

# Simulating received codeword with some bit flips
sended_codeword = random.choice(list(codewords_dict.values()))  # Select a random codeword
position = random.randint(0, n-1)  # Generate a random position
received_codeword = flip_bit(sended_codeword, position)  # Flip the bit at the random position
print("\n\nSended Codeword:", sended_codeword)
print("\nReceived Codeword:", received_codeword)

# Decoding
syndrome = ldpc_decode(received_codeword, parity_check_matrix)
print("\nSyndrome:", syndrome)

# Correcting
corrected_codeword = ldpc_correct(received_codeword, parity_check_matrix)
