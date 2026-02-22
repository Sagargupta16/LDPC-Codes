# LDPC Codes

Python implementation and simulation of Low-Density Parity-Check (LDPC) codes for error correction in digital communications.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)

## Overview

LDPC codes are a class of linear error-correcting codes that achieve near-Shannon-limit performance on a wide range of communication channels. This project provides a complete Python implementation of LDPC encoding, decoding, and simulation -- including applications for image and audio transmission over noisy channels.

## Features

- **LDPC Code Construction** -- Generate parity-check matrix (H) and generator matrix (G) with configurable parameters
- **Encoding & Decoding** -- Full encoding pipeline with belief propagation (Log-BP) decoding
- **BER Simulation** -- Bit Error Rate analysis across varying SNR levels
- **Image Transmission** -- Simulate sending images through noisy channels with LDPC protection
- **Audio Transmission** -- LDPC-coded audio transmission simulation
- **Parallel Decoding** -- Numba-accelerated parallel decoding implementation
- **Visualization** -- Matplotlib-based plots for BER curves and transmission results

## Project Structure

```
LDPC-Codes/
├── ldpc/                              # Core LDPC library
│   ├── __init__.py                    # Package exports
│   ├── code.py                        # LDPC code construction (make_ldpc)
│   ├── encoder.py                     # Encoding functions
│   ├── decoder.py                     # Log-BP decoding (with Numba acceleration)
│   ├── utils.py                       # General utilities
│   ├── utils_img.py                   # Image processing utilities
│   ├── utils_audio.py                 # Audio processing utilities
│   ├── ldpc_images.py                 # Image transmission simulation
│   ├── ldpc_audio.py                  # Audio transmission simulation
│   └── _version.py                    # Version information
├── binary/                            # Binary data files
├── data/                              # Sample data for simulations
├── plot_coding_decoding_simulation.py # BER vs SNR simulation script
├── plot_image_transmission.py         # Image transmission demo
├── plot_parallel_decoding.py          # Parallel decoding benchmark
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Numba (for parallel decoding acceleration)

### Installation

```bash
git clone https://github.com/Sagargupta16/LDPC-Codes.git
cd LDPC-Codes
pip install numpy scipy matplotlib numba
```

### Usage

#### Basic Coding & Decoding Simulation

```bash
python plot_coding_decoding_simulation.py
```

This generates a BER vs SNR curve showing error correction performance:
- Creates an LDPC code with `n=30` coded bits, `d_v=2`, `d_c=3`
- Simulates 50 transmissions per SNR point across SNR range [-2, 10] dB
- Plots the resulting Bit Error Rate curve

#### Image Transmission

```bash
python plot_image_transmission.py
```

Demonstrates LDPC-coded image transmission through a noisy AWGN channel with visual comparison of original vs. received images.

#### Parallel Decoding Benchmark

```bash
python plot_parallel_decoding.py
```

Benchmarks Numba-accelerated parallel decoding against standard sequential decoding.

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `n` | Number of coded bits | 30-1000+ |
| `d_v` | Column weight (ones per column in H) | 2-4 |
| `d_c` | Row weight (ones per row in H) | 3-6 |
| `snr` | Signal-to-Noise Ratio (dB) | -2 to 10 |
| `maxiter` | Maximum decoding iterations | 50-100 |

## API Reference

```python
from ldpc import make_ldpc, encode, decode, get_message

# Create LDPC code
H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)

# Encode message
y = encode(G, message, snr, seed=seed)

# Decode received signal
D = decode(H, y, snr, maxiter=50)

# Extract original message
x = get_message(G, D)
```

## License

MIT
