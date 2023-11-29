from PIL import Image, ImageEnhance
import numpy as np

def additional_enhancement(image,enhancement= 1.5):
    # Convert the image to a PIL Image if it's not already
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(enhancement)
    enhanced_image = np.array(enhanced_image)

    return enhanced_image



if __name__ == "__main__":
    file_path = "converted_image_8bpp_gray.png"
    original_image = Image.open(file_path)

    # Apply additional enhancement using PIL
    enhanced_image = additional_enhancement(original_image, enhancement=2)

    # Save the enhanced image using PIL
    Image.fromarray(enhanced_image).save("Enhanced_image.png")
    print("Enhanced image saved successfully!")
