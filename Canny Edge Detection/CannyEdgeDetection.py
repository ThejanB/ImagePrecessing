###############################################
### Canny Edge Detection using PIL and math ###
#################  200698X ####################
###############################################

import sys
import math
from PIL import Image

def rgb2gray(img):
    # Convert RGB image to grayscale
    width, height = img.size
    gray_image = []
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            row.append(gray)
        gray_image.append(row)
    return gray_image

def gaussian_kernel(size, sigma=1):
    # Create a size x size Gaussian kernel
    kernel = []
    sum_val = 0
    for x in range(size):
        row = []
        for y in range(size):
            x_val = x - size // 2
            y_val = y - size // 2
            exponent = -(x_val**2 + y_val**2) / (2 * sigma**2)
            value = (1 / (2 * math.pi * sigma**2)) * math.exp(exponent)
            row.append(value)
            sum_val += value
        kernel.append(row)
    # Normalize the kernel
    for x in range(size):
        for y in range(size):
            kernel[x][y] /= sum_val
    return kernel

def pad_image(image, pad_size, mode='edge'):
    height = len(image)
    width = len(image[0])
    new_height = height + 2 * pad_size
    new_width = width + 2 * pad_size
    padded_image = [[0 for _ in range(new_width)] for _ in range(new_height)]
    for i in range(height):
        for j in range(width):
            padded_image[i + pad_size][j + pad_size] = image[i][j]
    # Handle the edges
    if mode == 'edge':
        # Top and bottom
        for i in range(pad_size):
            padded_image[i][pad_size:-pad_size] = padded_image[pad_size][pad_size:-pad_size]
            padded_image[-(i+1)][pad_size:-pad_size] = padded_image[-(pad_size+1)][pad_size:-pad_size]
        # Left and right
        for i in range(new_height):
            for j in range(pad_size):
                padded_image[i][j] = padded_image[i][pad_size]
                padded_image[i][-(j+1)] = padded_image[i][-(pad_size+1)]
    return padded_image

def convolve(image, kernel):
    height = len(image)
    width = len(image[0])
    kernel_size = len(kernel)
    pad_size = kernel_size // 2
    padded_image = pad_image(image, pad_size, mode='edge')
    new_image = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            sum_val = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    sum_val += kernel[m][n] * padded_image[i + m][j + n]
            new_image[i][j] = sum_val
    return new_image

def sobel_filters(img):
    Kx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    Ky = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    height = len(img)
    width = len(img[0])
    G = [[0 for _ in range(width)] for _ in range(height)]
    theta = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            G[i][j] = math.hypot(Ix[i][j], Iy[i][j])
            theta[i][j] = math.atan2(Iy[i][j], Ix[i][j])
    # Normalize G to 0-255
    max_G = max(map(max, G))
    for i in range(height):
        for j in range(width):
            G[i][j] = G[i][j] / max_G * 255
    return (G, theta)

def non_max_suppression(G, theta):
    height = len(G)
    width = len(G[0])
    Z = [[0 for _ in range(width)] for _ in range(height)]
    angle = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            angle_deg = theta[i][j] * 180.0 / math.pi
            if angle_deg < 0:
                angle_deg += 180
            angle[i][j] = angle_deg
    for i in range(1, height-1):
        for j in range(1, width-1):
            q = 255
            r = 255
            # Angle 0
            if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
                q = G[i][j + 1]
                r = G[i][j - 1]
            # Angle 45
            elif (22.5 <= angle[i][j] < 67.5):
                q = G[i + 1][j - 1]
                r = G[i - 1][j + 1]
            # Angle 90
            elif (67.5 <= angle[i][j] < 112.5):
                q = G[i + 1][j]
                r = G[i - 1][j]
            # Angle 135
            elif (112.5 <= angle[i][j] < 157.5):
                q = G[i - 1][j - 1]
                r = G[i + 1][j + 1]
            if (G[i][j] >= q) and (G[i][j] >= r):
                Z[i][j] = G[i][j]
            else:
                Z[i][j] = 0
    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    height = len(img)
    width = len(img[0])
    max_val = max(map(max, img))
    highThreshold = max_val * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    res = [[0 for _ in range(width)] for _ in range(height)]
    weak = 25
    strong = 255
    for i in range(height):
        for j in range(width):
            if img[i][j] >= highThreshold:
                res[i][j] = strong
            elif img[i][j] >= lowThreshold:
                res[i][j] = weak
            else:
                res[i][j] = 0
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    height = len(img)
    width = len(img[0])
    for i in range(1, height-1):
        for j in range(1, width-1):
            if img[i][j] == weak:
                # Check if any neighbor is strong
                if (img[i+1][j-1]==strong or img[i+1][j]==strong or img[i+1][j+1]==strong
                    or img[i][j-1]==strong or img[i][j+1]==strong
                    or img[i-1][j-1]==strong or img[i-1][j]==strong or img[i-1][j+1]==strong):
                    img[i][j] = strong
                else:
                    img[i][j] = 0
    return img

def canny_edge_detection(img):

    # Step 1: Convert to grayscale
    img_gray = rgb2gray(img)

    # Step 2: Apply Gaussian Blur
    kernel_size = 5
    sigma = 1
    kernel = gaussian_kernel(kernel_size, sigma)
    img_blur = convolve(img_gray, kernel)

    # Step 3: Compute gradient intensity and direction
    G, theta = sobel_filters(img_blur)

    # Step 4: Non-maximum suppression
    img_nms = non_max_suppression(G, theta)

    # Step 5: Double threshold
    img_thresh, weak, strong = threshold(img_nms)

    # Step 6: Edge tracking by hysteresis
    img_final = hysteresis(img_thresh, weak, strong)

    # Step 7: Convert the final image to PIL Image for saving and displaying
    height, width = len(img_final), len(img_final[0])
    output_image = Image.new('L', (width, height))
    pixels = []
    for i in range(height):
        for j in range(width):
            pixels.append(int(img_final[i][j]))
    output_image.putdata(pixels)
    return output_image

def main():

    image_file = "input_image.jpg"
    try:
        img = Image.open(image_file)
    except IOError:
        print("Failed to load image.")
        sys.exit(1)

    edge_img = canny_edge_detection(img)

    output_file = image_file.rsplit('.',1)[0] + '_edge.png'
    edge_img.save(output_file)
    print("Edge image saved as", output_file)

if __name__ == "__main__":
    main()
