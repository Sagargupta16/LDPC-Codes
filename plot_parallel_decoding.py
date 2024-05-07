from time import time
import numpy as np
from matplotlib import pyplot as plt
from ldpc import make_ldpc, decode, encode

# Set parameters for LDPC code
n = 100
d_v = 2
d_c = 5
seed = 42
rng = np.random.RandomState(seed)

# Print information about LDPC code
print("Creating LDPC code...")
H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)

n, k = G.shape
print("Number of coded bits:", k)

# Simulate transmission for different levels of noise
print("Simulating transmission for different levels of noise...")
n_messages = np.arange(1, 20)
n_runs = 50
snr = 8
times_parallel = []
times_sequential = []

# Loop over different number of messages
for pp in n_messages:
    print("Number of messages:", pp)
    t_parallel = 0
    t_seq = 0
    V = rng.randint(2, size=(k, pp))  # Simulate messages
    
    Y = encode(G, V, snr, seed=seed)  # Encode messages with LDPC code
    
    # Parallel decoding
    for _ in range(n_runs):
        t = time()
        decode(H, Y, snr)  # Decode all messages in parallel
        t_parallel += time() - t

    # Sequential decoding
    for _ in range(n_runs):
        t = time()
        for y in Y.T:
            decode(H, y, snr)  # Decode each message sequentially
        t_seq += time() - t
    
    # Average times over runs and append to lists
    times_sequential.append(t_seq / n_runs)
    times_parallel.append(t_parallel / n_runs)

# Plot results
plt.figure()
plt.plot(n_messages, times_sequential, color="indianred", lw=2, label="Sequential")
plt.plot(n_messages, times_parallel, color="gold", lw=2, label="Parallel")
plt.ylabel("Time (s)")
plt.xlabel("# Messages")
plt.legend()
plt.title("Time Comparison for Sequential vs. Parallel Decoding")
plt.grid(True)
plt.show()
