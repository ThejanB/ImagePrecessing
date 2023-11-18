"""
200698X  - Image resampling assignment
due date - 2023/09/26

# Input file name should be "input.jpeg"

"""

import cv2
import numpy as np

def nearest_neighbor_interpolation(image, new_height, new_width):
    height, width, _ = image.shape
    x_scale = new_width / width
    y_scale = new_height / height

    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x / x_scale)
            src_y = int(y / y_scale)
            new_image[y, x] = image[src_y, src_x]

    return new_image

def bilinear_interpolation(image, new_height, new_width):
    height, width, _ = image.shape
    #print(height, width, _)

    x_scale = new_width / width
    y_scale = new_height / height

    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            src_x = x / x_scale
            src_y = y / y_scale

            x1, y1 = int(src_x), int(src_y)
            x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

            dx, dy = src_x - x1, src_y - y1

            pixel1 = image[y1, x1]
            pixel2 = image[y1, x2]
            pixel3 = image[y2, x1]
            pixel4 = image[y2, x2]

            new_pixel = (
                (1 - dx) * (1 - dy) * pixel1 +
                dx * (1 - dy) * pixel2 +
                (1 - dx) * dy * pixel3 +
                dx * dy * pixel4
            )

            new_image[y, x] = new_pixel.astype(np.uint8)

    return new_image


# Load the image using OpenCV
input_image = cv2.imread("input.jpeg")

if input_image is None:
    print("Input image file could not be found.")

else:
    # Define the new dimensionsss
    new_height = 300
    new_width = 400

    # Perform nearest neighbor interpolation
    nearest_neighbor_result = nearest_neighbor_interpolation(input_image, new_height, new_width)

    # Perform bilinear interpolation
    bilinear_result = bilinear_interpolation(input_image, new_height, new_width)

    # Save the results using OpenCV
    cv2.imwrite("nearest_neighbor_result.jpg", nearest_neighbor_result)
    print("nearest_neighbor_result.jpg saved.")
    cv2.imwrite("bilinear_result.jpg", bilinear_result)
    print("nearest_neighbor_result.jpg saved.")

print("Done.")
