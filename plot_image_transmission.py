from ldpc import code, ldpc_images
from ldpc.utils_img import gray2bin, rgb2bin

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from time import time

def process_image(image_path, coding_matrix, snr, seed, togray=False):
    print("Processing", image_path)
    # Load image
    if togray:
        # Convert image to grayscale and then binary
        image = np.asarray(Image.open(image_path).convert('L'))
    else:
        # Convert image from RGB to binary
        image = np.asarray(Image.open(image_path))

    print("Image shape:", image.shape)

    # Convert image to binary
    image_bin = gray2bin(image) if togray else rgb2bin(image)
    print("Binary image shape:", image_bin.shape)

    # Encode image
    start_time = time()
    coded_image, noisy_image = ldpc_images.encode_img(coding_matrix, image_bin, snr, seed=seed)
    encoding_time = time() - start_time

    print("Coded image shape:", coded_image.shape)

    # Decode image
    start_time = time()
    decoded_image = ldpc_images.decode_img(coding_matrix, H, coded_image, snr, image_bin.shape)
    decoding_time = time() - start_time

    print("Decoded image shape:", decoded_image.shape)

    # Calculate errors
    error_noisy = abs(noisy_image - image).mean()
    error_decoded = abs(decoded_image - image).mean()

    print("Noisy image error: %.3f %%" % error_noisy)
    print("Decoded image error: %.3f %%" % error_decoded)
    print("Encoding time: %.3f seconds" % encoding_time)
    print("Decoding time: %.3f seconds" % decoding_time)

    print("Processing completed for", image_path)
    return image, noisy_image, decoded_image, error_noisy, error_decoded, encoding_time, decoding_time


n = 200
d_v = 3
d_c = 4
seed = 42
snr = 9

H, G = code.make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)

eye_path = "./data/eye.png"
tiger_path = "./data/tiger.jpg"

eye_results = process_image(eye_path, G, snr, seed, togray=True)
print("\n\n")
tiger_results = process_image(tiger_path, G, snr, seed, togray=False)

def plot_images(all_images, all_titles):
    num_rows = len(all_images)
    num_cols = len(all_images[0])

    f, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))
    for i, (row_images, row_titles) in enumerate(zip(all_images, all_titles)):
        for j, (image, title) in enumerate(zip(row_images, row_titles)):
            ax = axes[i, j]
            ax.imshow(image, cmap='gray')
            ax.set_title(title, fontsize=16)
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Usage example:
titles_eye = ["Original", "Noisy | Err = %.3f %%" % eye_results[3],
              "Decoded | Err = %.3f %%" % eye_results[4]]
titles_tiger = ["Original", "Noisy | Err = %.3f %%" % tiger_results[3],
                "Decoded | Err = %.3f %%" % tiger_results[4]]
all_imgs = [[eye_results[0], eye_results[1], eye_results[2]], 
            [tiger_results[0], tiger_results[1], tiger_results[2]]]
all_titles = [titles_eye, titles_tiger]

plot_images(all_imgs, all_titles)

