import numpy as np
from ldpc import make_ldpc, decode, get_message, encode
from matplotlib import pyplot as plt

# Set parameters
n = 30 # number of bits
d_v = 2 # number of ones in each column of H
d_c = 3 # number of ones in each row of H
seed = np.random.RandomState(42)

# Print information about LDPC code
print("Creating LDPC code...")
H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
n, k = G.shape
print("Number of coded bits:", k)

# Simulation parameters
snrs = np.linspace(-2, 10, 25)  # signal to noise ratio
v = np.arange(k) % 2  # fixed k bits message
n_trials = 50  # number of transmissions with different noise
V = np.tile(v, (n_trials, 1)).T  # stack v in columns

# Simulation and decoding
errors = []
for snr in snrs:
    print("Simulating transmission for SNR:", snr)
    y = encode(G, V, snr, seed=seed)
    D = decode(H, y, snr, maxiter=50) 
    error = 0.
    for i in range(n_trials):
        x = get_message(G, D[:, i])
        error += abs(v - x).sum() / (k * n_trials)
    errors.append(error)

# Plot results
plt.figure()
plt.plot(snrs, errors, color="indianred")
plt.ylabel("Bit error rate")
plt.xlabel("SNR")
plt.title("LDPC Coding and Decoding Simulation")
plt.grid(True)
plt.show()
