from matplotlib import pyplot as plt
import numpy as np

def plot_histogram(image,save_path="Final-histogram_plot.png"):
    image_array = np.array(image)

    plt.hist(image_array.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Final Image')

    # Save the histogram plot
    plt.savefig(save_path)
    print("Histogram plot saved successfully!")

    # Display the histogram plot
    plt.show()