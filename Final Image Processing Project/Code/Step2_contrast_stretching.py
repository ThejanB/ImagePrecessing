from PIL import Image
import numpy as np

def linear_contrast_stretching(image):
    img_array = np.array(image)
    
    # Compute minimum and maximum pixel values
    min_intensity = np.min(img_array)
    max_intensity = np.max(img_array)
    
    # Apply linear contrast stretching to the entire image
    stretched_img = ((img_array - min_intensity) / (max_intensity - min_intensity)) * 255
    stretched_img = stretched_img.astype(np.uint8)
    
    return Image.fromarray(stretched_img)

def histogram_equalization(image):
    img_array = np.array(image)
    
    # Compute histogram and cumulative distribution function (CDF)
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())
    
    # Map each pixel's value to its new equalized value
    equalized_img = np.interp(img_array.flatten(), bins[:-1], cdf_normalized * 255).reshape(img_array.shape)
    
    return Image.fromarray(equalized_img.astype(np.uint8))

def intensity_level_slicing(image, low_threshold = 30, high_threshold = 220):
    img_array = np.array(image)
    
    # Define a mask to identify pixels within the specified intensity range
    mask = (img_array >= low_threshold) & (img_array <= high_threshold)
    
    sliced_img = np.zeros_like(img_array)
    sliced_img[mask] = img_array[mask]
    
    return Image.fromarray(sliced_img.astype(np.uint8))

if __name__ == "__main__":
    inputPath = "converted_image_8bpp_gray.png"

    # Load the image
    original_image = Image.open(inputPath)

    # Apply linear contrast stretching and Save the resulting image
    stretched_image = linear_contrast_stretching(original_image)
    stretched_image.save("CS-histogram_equalized_image.png")
    print("Linear Contrast Stretched histogram_equalized Image saved successfully!")
    
    # Apply histogram equalization and Save the resulting image
    equalized_image = histogram_equalization(original_image)
    equalized_image.save("CS-linear_contrast_stretching_img.png")
    print("Linear Contrast Stretched histogram_equalized Image saved successfully!")
    
    # Apply intensity level slicing to the image and Save the resulting image
    sliced_image = intensity_level_slicing(original_image, 30, 220)
    sliced_image.save("CS-intensity_sliced_image.png")
    print("Linear Contrast Stretched Image saved successfully!")
