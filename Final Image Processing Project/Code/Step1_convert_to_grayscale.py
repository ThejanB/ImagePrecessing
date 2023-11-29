from PIL import Image

def convert_to_grayscale(file_path):

    img = Image.open(file_path)

    # Convert to RGB mode if the image is not in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size
    gray_img = Image.new('L', (width, height))

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            gray_value = int(r * 0.2989 + g * 0.587 + b * 0.114)
            gray_img.putpixel((x, y), gray_value)

    return gray_img

if __name__ == "__main__":

    # Apply grayscale conversion and save the resulting image
    file_path = "SampleImage.png"
    gray_image = convert_to_grayscale(file_path)
    gray_image.save("converted_image_8bpp_gray.png")
    print("Gray Filtered Images saved successfully!")