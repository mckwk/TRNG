import cv2
import numpy as np

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
    # Load the color image
    color_image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = grayscale(color_image)

    # Apply error diffusion dithering
    binary_image = error_diffusion_dithering(gray_image)

    return binary_image


import numpy as np

def arnold_cat_map(image, iterations=7):
    height, width = image.shape

    # Reshape the image to a 1D array
    image_flat = image.flatten()

    # Set parameters
    p = 1
    q = 1
    N = 512

    # Generate the permutation matrix
    permutation_matrix = np.array([[1, p], [q, (p*q + 1) % N]])

    # Iterate the Arnold cat map
    for _ in range(iterations):
        # Reshape the flattened image back to 2D
        image_flat = image_flat.reshape((height, width))

        # Create coordinates grid
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Apply the permutation to the image coordinates
        new_coords = (permutation_matrix @ np.array([x.flatten(), y.flatten()])) % height

        # Flatten the coordinates and index the image with them
        image_flat = image_flat[new_coords[1], new_coords[0]]

    # Reshape the flattened image back to the original shape
    shuffled_image = image_flat.reshape((height, width))

    return shuffled_image


# Example usage
# Assuming binary_image is the binary image obtained from the previous steps
# Example usage
binary_image = transform_to_binary("lena.png")


shuffled_image = arnold_cat_map(binary_image)
cv2.imshow("Shuffled Image", shuffled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()