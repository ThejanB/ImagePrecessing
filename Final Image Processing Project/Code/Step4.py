import os
import numpy as np

def compute_entropy(image):
    # Convert the image to a NumPy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    probabilities = hist / float(np.sum(hist))
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_value

def compute_compression_ratio(original_file_path, compressed_file_path):
    # Get file sizes
    original_size = os.path.getsize(original_file_path)
    compressed_size = os.path.getsize(compressed_file_path)

    # Compute compression ratio
    compression_ratio = original_size / compressed_size
    return compression_ratio