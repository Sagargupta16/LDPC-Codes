import numpy as np
from scipy.sparse import csr_matrix
from . import utils

def parity_check_matrix(n, dv, dc, seed=None):
    """
    Generate a regular Parity-Check Matrix H following Gallager's algorithm.

    Parameters
    ----------
    n: int
        Length of the codewords.
    dv: int
        Number of parity-check equations including a certain bit.
    dc: int
        Number of bits in the same parity-check equation. Must be greater or equal to dv and divide n.
    seed: int, optional
        Seed of the random generator.

    Returns
    -------
    H: array
        LDPC regular matrix H with dimensions (n_equations, n_code).
    """
    rng = utils.check_random_state(seed)

    if dv <= 1 or dc <= dv or n % dc:
        raise ValueError("Invalid parameters.")

    n_eq = n * dv // dc
    block = np.zeros((n_eq // dv, n), dtype=int)
    H = np.empty((n_eq, n))

    # Fill the first block with consecutive ones in each row
    block_size = n_eq // dv
    for i in range(block_size):
        for j in range(i * dc, (i+1) * dc):
            block[i, j] = 1
    H[:block_size] = block

    # Create remaining blocks by permutations of the first block's columns
    for i in range(1, dv):
        H[i * block_size: (i + 1) * block_size] = rng.permutation(block.T).T

    return H.astype(int)

def coding_matrix(H, sparse=True):
    """
    Compute the generating coding matrix G given the LDPC matrix H.

    Parameters
    ----------
    H: array
        Parity check matrix of an LDPC code.
    sparse: bool, optional
        If True, scipy.sparse format is used to speed up computation.

    Returns
    -------
    G.T: array
        Transposed coding matrix.
    """
    if type(H) == csr_matrix:
        H = H.toarray()

    # Apply Gaussian elimination
    Href_colonnes, tQ = utils.gaussjordan(H.T, 1)
    Href_diag = utils.gaussjordan(np.transpose(Href_colonnes))
    Q = tQ.T
    n_bits = H.shape[1] - Href_diag.sum()
    Y = np.zeros(shape=(H.shape[1], n_bits)).astype(int)
    Y[H.shape[1] - n_bits:, :] = np.identity(n_bits)

    if sparse:
        Q, Y = csr_matrix(Q), csr_matrix(Y)

    tG = utils.binaryproduct(Q, Y)

    return tG

def coding_matrix_systematic(H, sparse=True):
    """
    Compute a coding matrix G in systematic format with an identity block.

    Parameters
    ----------
    H: array
        Parity-check matrix.
    sparse: bool, optional
        If True, scipy.sparse is used to speed up computation if n_code > 1000.

    Returns
    -------
    H_new: array
        Modified parity-check matrix.
    G_systematic.T: array
        Transposed Systematic Coding matrix associated to H_new.
    """
    n_eq, n_code = H.shape

    # Determine if sparse matrices should be used
    sparse = n_code > 1000 or sparse

    # Initialize identity matrix
    P1 = np.identity(n_code, dtype=int)

    # Apply Gaussian elimination to transform H into systematic form
    Hrowreduced = utils.gaussjordan(H)
    n_bits = n_code - sum([a.any() for a in Hrowreduced])

    while True:
        zeros = [i for i in range(min(n_eq, n_code)) if not Hrowreduced[i, i]]
        if zeros:
            indice_colonne_a = min(zeros)
        else:
            break
        list_ones = [j for j in range(indice_colonne_a + 1, n_code) if Hrowreduced[indice_colonne_a, j]]
        if list_ones:
            indice_colonne_b = min(list_ones)
        else:
            break
        Hrowreduced[:, [indice_colonne_a, indice_colonne_b]] = Hrowreduced[:, [indice_colonne_b, indice_colonne_a]]
        P1[:, [indice_colonne_a, indice_colonne_b]] = P1[:, [indice_colonne_b, indice_colonne_a]]

    P1 = P1.T
    identity = list(range(n_code))
    sigma = identity[n_code - n_bits:] + identity[:n_code - n_bits]
    P2 = np.zeros(shape=(n_code, n_code), dtype=int)
    P2[identity, sigma] = np.ones(n_code)

    if sparse:
        P1, P2, H = csr_matrix(P1), csr_matrix(P2), csr_matrix(H)

    P = utils.binaryproduct(P2, P1)
    H_new = utils.binaryproduct(H, np.transpose(P))

    G_systematic = np.zeros((n_bits, n_code), dtype=int)
    G_systematic[:, :n_bits] = np.identity(n_bits)
    G_systematic[:, n_bits:] = (Hrowreduced[:n_code - n_bits, n_code - n_bits:]).T

    return H_new, G_systematic.T

def make_ldpc(n, dv, dc, systematic=False, sparse=True, seed=None):
    """
    Create an LDPC coding and decoding matrices H and G.

    Parameters
    ----------
    n: int
        Length of the codewords.
    dv: int
        Number of parity-check equations including a certain bit.
    dc: int
        Number of bits in the same parity-check equation.
    systematic: bool, optional
        If True, constructs a systematic coding matrix G.
    sparse: bool, optional
        If True, scipy.sparse format is used to speed up computation.
    seed: int, optional
        Seed of the random generator.

    Returns
    -------
    H: array
        Parity check matrix.
    G: array
        Coding matrix.
    """
    seed = utils.check_random_state(seed)
    H = parity_check_matrix(n, dv, dc, seed=seed)
    if systematic:
        H, G = coding_matrix_systematic(H, sparse=sparse)
    else:
        G = coding_matrix(H, sparse=sparse)
    return H, G
