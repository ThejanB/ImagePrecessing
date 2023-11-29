import numpy as np
from PIL import Image

def mean_filter(image, filter_size = 3):
    width, height = image.size
    img_array = np.array(image)
    
    kernel = np.ones((filter_size, filter_size)) / (filter_size * filter_size)
    pad_width = filter_size // 2
    padded_image = np.pad(img_array, pad_width, mode='edge')
    filtered_image = np.zeros_like(img_array)
    
    # Apply mean filter
    for y in range(height):
        for x in range(width):
            roi = padded_image[y : y + filter_size, x : x + filter_size]
            filtered_value = np.sum(roi * kernel)
            filtered_image[y, x] = int(filtered_value)
    
    return Image.fromarray(filtered_image)

def median_filter(image, filter_size = 3):

    # width and height of the image
    width, height = image.size
    # print(width, height)
    img_array = np.array(image)
    
    pad_width = filter_size // 2
    padded_image = np.pad(img_array, pad_width, mode='edge')
    
    filtered_image = np.zeros_like(img_array)
    
    for y in range(height):
        for x in range(width):
            roi = padded_image[y : y + filter_size, x : x + filter_size]
            
            median_value = np.median(roi)
            filtered_image[y, x] = int(median_value)
    
    return Image.fromarray(filtered_image)

if __name__ == "__main__":

    # Define the variables
    filter_size = 3
    file_path = "converted_image_8bpp_gray.png"

    # Load the image
    original_image = Image.open(file_path)

    # Apply the mean filter and Save the resulting image
    median_filtered_image = mean_filter(original_image, filter_size)
    median_filtered_image.save("NF-mean_filtered_image.png")
    print("Mean Filtered Images saved successfully!")

    # Apply the median filter and Save the resulting image
    median_filtered_image = median_filter(original_image, filter_size)
    median_filtered_image.save("NF-median_filtered_image.png")
    print("Median Filtered Images saved successfully!")
    
  
