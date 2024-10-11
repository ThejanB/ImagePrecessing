##################################################
### Canny Edge Detection with Numpy and OpenCV ###
####################  200698X ####################
##################################################

import sys
import numpy as np
import cv2

def rgb2gray(img):
    # Convert BGR to Grayscale using the luminosity method
    R = img[:,:,2]
    G = img[:,:,1]
    B = img[:,:,0]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray.astype(np.float32)

def gaussian_kernel(size, sigma=1):
    # Create a (size x size) Gaussian kernel
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)  # Normalize the kernel

def convolve(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h //2
    pad_w = kernel_w //2
    
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge') # Pad the image
    
    new_image = np.zeros_like(image)    # Initialize output image

    # Convolve
    for i in range(image_h):
        for j in range(image_w):
            new_image[i,j] = np.sum(kernel * padded_image[i:i+kernel_h, j:j+kernel_w])
    return new_image

def sobel_filters(img):     # I used Sobel filters to compute the gradient intensity and direction and reduce noise.
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255  # Normalize to 0-255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)

def non_max_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle<0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = G[i+1, j]
                    r = G[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]
                if (G[i,j] >= q) and (G[i,j] >= r):
                    Z[i,j] = G[i,j]
                else:
                    Z[i,j] = 0
            except IndexError as e:
                pass
    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
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
    return img_final.astype(np.uint8)

def main():
    image_file = "input_image.jpg"
    img = cv2.imread(image_file)
    if img is None:
        print("Failed to load image.")
        sys.exit(1)

    edge_img = canny_edge_detection(img)

    output_file_name = image_file.rsplit('.',1)[0] + '_edge.png'
    cv2.imwrite(output_file_name, edge_img)
    print(f"Edge image saved as {output_file_name}")

if __name__ == "__main__":
    main()

