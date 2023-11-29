from PIL import Image, ImageFilter
import numpy as np

def unsharp_masking(image, sigma=1.0, strength=1.5):
    img_array = np.array(image)

    if len(img_array.shape) == 3:
        img_array = np.array(image.convert("L"))
    
    # Applying Gaussian blur to the image
    blurred_img = image.filter(ImageFilter.GaussianBlur(sigma))
    blurred_array = np.array(blurred_img)
    high_pass = img_array - blurred_array
    sharpened_array = img_array + strength * high_pass
    
    sharpened_array = np.clip(sharpened_array, 0, 255)
    sharpened_Image = Image.fromarray(sharpened_array.astype(np.uint8))
    return sharpened_Image


def wiener_deconvolution(blurred_image, psf = np.ones((5, 5)) / 25, noise_var = 0.1):
    blurred_array = np.array(blurred_image)
    
    psf_fft = np.fft.fft2(psf, blurred_array.shape)
    blurred_fft = np.fft.fft2(blurred_array)
    wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft)**2 + noise_var)
    deconvolved_fft = blurred_fft * wiener_filter
    
    deconvolved_array = np.fft.ifft2(deconvolved_fft)
    deconvolved_array = np.abs(deconvolved_array)
    deconvolved_Image = Image.fromarray(deconvolved_array.astype(np.uint8))
    return deconvolved_Image

if __name__ == "__main__":
    input_image_path = "converted_image_8bpp_gray.png"
    sigma_val = 1.0  # Gaussian blur sigma value
    strength_val = 1.5  # Strength of the sharpening effect

    # Load the image
    blurred_image = Image.open(input_image_path)
    
    # Apply unsharp masking to the image
    sharpened_image = unsharp_masking(blurred_image, sigma=sigma_val, strength=strength_val)
    sharpened_image.save("D-unsharp_masked_image.png")
    print("Unsharp Masked Image saved successfully!")

    # Define or estimate the PSF and noise variance (for demonstration)
    estimated_psf = np.ones((5, 5)) / 25
    estimated_noise_var = 0.1
    
    # Apply Wiener deconvolution to the blurred image
    deconvolved_image = wiener_deconvolution(blurred_image, estimated_psf, estimated_noise_var)
    deconvolved_image.save("D-deconvolved_image.png")
    print("Deconvolved Image saved successfully!")
