import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def error_diffusion_dithering(image):
    height, width = image.shape
    binary_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            binary_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x < width - 1:
                image[y, x + 1] += quant_error * 7 / 16
            if y < height - 1:
                if x > 0:
                    image[y + 1, x - 1] += quant_error * 3 / 16
                image[y + 1, x] += quant_error * 5 / 16
                if x < width - 1:
                    image[y + 1, x + 1] += quant_error * 1 / 16

    return binary_image


def transform_to_binary(image_path):
    color_image = cv2.imread(image_path)
    gray_image = grayscale(color_image)
    binary_image = error_diffusion_dithering(gray_image)

    return binary_image


def arnold_cat_map(image, iterations=7):  # post-processing
    height, width = image.shape
    image_flat = image.flatten()

    # Set parameters
    p = 1
    q = 1
    N = 4000

    permutation_matrix = np.array([[1, p], [q, (p * q + 1) % N]])

    # Iterate the Arnold cat map
    for _ in range(iterations):
        image_flat = image_flat.reshape((height, width))
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Apply the permutation to the image coordinates
        new_coords = (
                             permutation_matrix @ np.array([x.flatten(), y.flatten()])
                     ) % height

        image_flat = image_flat[new_coords[1], new_coords[0]]

    shuffled_image = image_flat.reshape((height, width))

    return shuffled_image


def divide_into_blocks(image):
    height, width = image.shape
    blocks = []
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            block = image[y: y + 4, x: x + 4]
            blocks.append(block)
    return blocks


def count_black_pixels(block):
    return np.sum(block)


def generate_random_sequence(blocks):
    sequence = []
    for block in blocks:
        black_pixel_count = count_black_pixels(block)
        value = 0 if black_pixel_count % 2 == 0 else 1
        sequence.append(value)
    return sequence


def zigzag_scan(blocks):
    sequence = []
    num_blocks = len(blocks)
    for i in range(num_blocks):
        if i % 2 == 0:
            sequence.append(blocks[i])
        else:
            sequence.append(int(not blocks[i]))
    return sequence


def entropy(probabilities):
    probabilities[probabilities == 0] = 1
    return -np.sum(probabilities * np.log2(probabilities))


def process_image(image_path):
    binary_image = transform_to_binary(image_path)
    binary_image_str = ''.join(str(pixel) for pixel in binary_image.flatten()).replace("255", "1")
    shuffled_image = arnold_cat_map(binary_image)
    blocks = divide_into_blocks(shuffled_image)
    random_sequence = generate_random_sequence(blocks)
    zigzag_sequence = zigzag_scan(random_sequence)
    random_sequence_array = np.array(zigzag_sequence)
    

    with open("binary_image.txt", "a") as file:
        file.write(binary_image_str)
    
    with open("random_sequence_array.txt", "a") as file:
        file.write(''.join(str(pixel) for pixel in random_sequence_array))

def divide_into_8bit(file):
    with open(file, 'r') as file:
        binary_sequence = file.read().strip()

    # Check if the length of the sequence is divisible by 8
    if len(binary_sequence) % 8 != 0:
        raise ValueError("Sequence length is not divisible by 8.")

    # Divide the sequence into 8-bit segments and convert each segment to an integer
    eight_bit_integers = [
        int(binary_sequence[i:i+8], 2) for i in range(0, len(binary_sequence), 8)
    ]

    return eight_bit_integers

def histograms(extractor_array, post_processed_array):
    plt.hist(post_processed_array, bins=range(256), color="blue", density=True)
    plt.title("Histogram of Eight-bit Numbers")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.show()

    plt.hist(extractor_array, bins=range(256), color="blue", density=True)
    plt.title("Histogram after extractor")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.show()


def singleProcessing(image):
    process_image(image)
    extractor_array = divide_into_8bit("binary_image.txt")
    post_processed_array = divide_into_8bit("random_sequence_array.txt")
    entropy_value = entropy(np.histogram(post_processed_array, bins=range(256), density=True)[0])
    print("Entropy:", entropy_value)

    print("Post-processed array length:" , len(post_processed_array))
    histograms(extractor_array, post_processed_array)

def multiProcessing(dir):
    file_paths = [os.path.join(dir, f"{i}.jpg") for i in range(1, 9)]
    # Process each image
    for file_path in file_paths:
        process_image(file_path)
    extractor_array = divide_into_8bit("binary_image.txt")
    post_processed_array = divide_into_8bit("random_sequence_array.txt")
    entropy_value = entropy(np.histogram(post_processed_array, bins=range(256), density=True)[0])
    print("Entropy:", entropy_value)

    print("Post-processed array length:" , len(post_processed_array))
    histograms(extractor_array, post_processed_array)

def main():
    if (os.path.exists("binary_image.txt")):
        os.remove("binary_image.txt")
    if (os.path.exists("random_sequence_array.txt")):
        os.remove("random_sequence_array.txt")

    start_time = time.time()
    #singleProcessing("kot.jpg")
    multiProcessing(os.path.join(os.getcwd(), "4000") )
    stop_time = time.time()

    duration = stop_time - start_time
    print("Czas trwania:", duration, "sekund")


if __name__ == "__main__":
    main()
