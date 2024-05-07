from ldpc import code, ldpc_images, utils_img

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from time import time


def process_image(image_path, coding_matrix, snr, seed):
    print("Processing", image_path)
    # Convert image from RGB to binary
    image = np.asarray(Image.open(image_path))

    print("Image shape:", image.shape)

    # Convert image to binary
    image_bin = utils_img.rgb2bin(image)
    print("Binary image shape:", image_bin.shape)

    # Encode image
    start_time = time()
    coded_image, noisy_image = ldpc_images.encode_img(
        coding_matrix, image_bin, snr, seed=seed
    )
    encoding_time = time() - start_time

    print("Coded image shape:", coded_image.shape)

    # Decode image
    start_time = time()
    decoded_image = ldpc_images.decode_img(
        coding_matrix, H, coded_image, snr, image_bin.shape
    )
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
    print("\n")
    return (
        image,
        noisy_image,
        decoded_image,
        error_noisy,
        error_decoded,
        encoding_time,
        decoding_time,
    )


def arrange_images(images_path):
    images_titles = []
    all_imgs = []

    for image_path in images_path:
        image_results = process_image(image_path, G, snr, seed)
        title = [
            "Original",
            "Noisy | Err = %.3f %%" % image_results[3],
            "Decoded | Err = %.3f %%" % image_results[4],
        ]
        images_titles.append(title)
        all_imgs.append([image_results[0], image_results[1], image_results[2]])

    return images_titles, all_imgs

def plot_images(all_images, all_titles):
    num_rows = len(all_images)
    num_cols = len(all_images[0])

    f, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    for i, (row_images, row_titles) in enumerate(zip(all_images, all_titles)):
        for j, (image, title) in enumerate(zip(row_images, row_titles)):
            ax = axes[i, j]
            ax.imshow(image, cmap="gray")
            ax.set_title(title, fontsize=16)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


n = 200 # Image size
d_v = 6 # Number of ones in each column
d_c = 10 # Number of ones in each row
seed = 42 # Seed for random number generator
snr = 8 # Signal to noise ratio

H, G = code.make_ldpc(n, d_v, d_c, seed=seed, sparse=True)

images_path = ["./data/eye.png", "./data/tiger.jpg"]

images_titles, all_imgs = arrange_images(images_path)

# Plot images

plot_images(all_imgs, images_titles)
