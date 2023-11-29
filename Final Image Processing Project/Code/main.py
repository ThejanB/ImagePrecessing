import os
import numpy as np
from PIL import Image

# filter functions
from Step1_convert_to_grayscale import convert_to_grayscale
from Step2_NoiseFilters import mean_filter, median_filter
from Step2_contrast_stretching import linear_contrast_stretching, histogram_equalization, intensity_level_slicing
from Step2_deblurring_filters import unsharp_masking, wiener_deconvolution as deblurring
from Step2_additional_enhancements import additional_enhancement
from Step3_plot_histogram import plot_histogram
from Step4 import compute_entropy, compute_compression_ratio

# Calculate Signal-to-Noise Ratio (SNR) using PIL
def calculate_snr(original_image, processed_image):
    original_array = np.array(original_image)
    processed_array = np.array(processed_image)

    mse = np.mean((original_array - processed_array) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel_value = np.max(original_array)
    snr = 10 * np.log10((max_pixel_value ** 2) / mse)
    
    return snr


if __name__ == "__main__":

    inputImagePath = "SampleImage.png"

    ###### Step 1 ######
    original_image = convert_to_grayscale(inputImagePath)
    original_image.save("grayImage.png")
    print("Gray Filtered Images saved successfully!")

    ###### Step 2 ######
    # Apply noise filters
    mean_filtered_image = mean_filter(original_image,5)
    mean_filtered_image.save("NF-mean_filtered_image.png")
    print("Mean Filtered Image saved successfully!")
    median_filtered_image = median_filter(original_image)
    median_filtered_image.save("NF-median_filtered_image.png")
    print("Median Filtered Image saved successfully!")

    # Apply contrast stretching
    stretched_image1 = linear_contrast_stretching(mean_filtered_image)
    stretched_image1.save("CS-linear_contrast_stretching_img.png")
    print("Linear Contrast Stretched Image1 saved successfully!")

    stretched_image2 = histogram_equalization(mean_filtered_image)
    stretched_image2.save("CS-histogram_equalized_image.png")
    print("Histogram_equalized Contrast Stretched Image2 saved successfully!")

    stretched_image3 = intensity_level_slicing(mean_filtered_image)
    stretched_image3.save("CS-intensity_sliced_image.png")
    print("Intensity_sliced Contrast Stretched Image3 saved successfully!")

    # Apply deblurring
    deblurred_image1 = deblurring(stretched_image1)
    deblurred_image2 = deblurring(stretched_image2)
    deblurred_image3 = deblurring(stretched_image3)

    # Apply additional enhancement
    enhanced_image1 = additional_enhancement(deblurred_image1)
    enhanced_image2 = additional_enhancement(deblurred_image2)
    enhanced_image3 = additional_enhancement(deblurred_image3)

    # Calculate SNR for each step and each image
    snr_mean_filtered = calculate_snr(original_image, mean_filtered_image)
    snr_median_filtered = calculate_snr(original_image, median_filtered_image)

    snr_stretched1 = calculate_snr(original_image, stretched_image1)
    snr_stretched2 = calculate_snr(original_image, stretched_image2)
    snr_stretched3 = calculate_snr(original_image, stretched_image3)

    snr_deblurred1 = calculate_snr(original_image, deblurred_image1)
    snr_deblurred2 = calculate_snr(original_image, deblurred_image2)
    snr_deblurred3 = calculate_snr(original_image, deblurred_image3)

    snr_enhanced1 = calculate_snr(original_image, enhanced_image1)
    snr_enhanced2 = calculate_snr(original_image, enhanced_image2)
    snr_enhanced3 = calculate_snr(original_image, enhanced_image3)

    ###### Step 3 ######
    # Find the image with the highest SNR
    if snr_enhanced1 > snr_enhanced2 and snr_enhanced1 > snr_enhanced3:
        enhanced_image = enhanced_image1
        snr_enhanced = snr_enhanced1
        snr_filtered = snr_mean_filtered
        snr_stretched = snr_stretched1
        snr_deblurred = snr_deblurred1
        print("Image 1 has the highest SNR")
    elif snr_enhanced2 > snr_enhanced1 and snr_enhanced2 > snr_enhanced3:
        enhanced_image = enhanced_image2
        snr_enhanced = snr_enhanced2
        snr_filtered = snr_mean_filtered
        snr_stretched = snr_stretched2
        snr_deblurred = snr_deblurred2
        print("Image 2 has the highest SNR")	
    else:
        enhanced_image = enhanced_image3
        snr_enhanced = snr_enhanced3
        snr_filtered = snr_mean_filtered
        snr_stretched = snr_stretched3
        snr_deblurred = snr_deblurred3
        print("Image 3 has the highest SNR")

    Image.fromarray(enhanced_image).save("Final-enhanced_image.png")
    print("Final Enhanced Image saved successfully!")
    
    #plot the histogram of the final image (which has the highest SNR)
    plot_histogram(enhanced_image) 

    ###### Step 4 ######
    entropy_value = compute_entropy(enhanced_image)
    compression_ratio = compute_compression_ratio("grayImage.png", "Final-enhanced_image.png")


    ###### Print the results ######
    print('\n------------------------------------------------------------')
    print('200698X Final image SNR values ->')
    print('------------------------------------------------------------\n')

    print(f"\tFiltered Image: {snr_filtered}")
    print(f"\tContrast Stretched Image: {snr_stretched}")
    print(f"\tDeblurred Image: {snr_deblurred}")
    print(f"\tEnhanced Image: {snr_enhanced}")

    print("\n\tEntropy:", entropy_value)
    print("\tCompression Ratio:", compression_ratio)
    print('\n------------------------------------------------------------')

    print("\nDone!")

