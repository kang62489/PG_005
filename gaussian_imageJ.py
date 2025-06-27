import os
import numpy as np
import imageio.v3 as imageio
from rich.progress import track
import imagej
import scyjava

def load_multipage_tiff(filename):
    """Load a multipage TIFF file into a numpy array."""
    print(f"Loading multipage TIFF: {filename}")
    return imageio.imread(filename)


def process_tiff_with_imagej_gaussian(input_file, output_folder, sigma=2.0):
    """Process a multipage TIFF file with ImageJ's Gaussian blur."""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Initialize ImageJ
    print("Initializing ImageJ...")
    ij = imagej.init('net.imagej:imagej', mode='headless')
    
    # Load the multipage TIFF
    images = load_multipage_tiff(input_file)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    print(f"Processing {len(images)} frames with Gaussian sigma={sigma}")
    
    # Process each frame
    final_result = []
    for i, img in enumerate(track(images, description="Processing frames")):
        # Convert numpy array to ImageJ image
        ij_img = ij.py.to_java(img)
        
        # Apply Gaussian blur using ImageJ
        # Get the op service and apply the filter
        blurred = ij.op().filter().gauss(ij_img, sigma)
        
        # Convert back to numpy array
        result_img = ij.py.from_java(blurred)
        final_result.append(result_img)
        
    
    # Save the processed image
    
    output_tiff = os.path.join(output_folder, f"{base_filename}_blurred.tif")
    imageio.imwrite(output_tiff, final_result)
    
    print(f"Processing complete. Results saved to {output_folder}")
    
    # Explicitly close the JVM when done
    scyjava.shutdown_jvm()


def main():
    # Get user input
    input_file = input("Enter the path to the multipage TIFF file: ")
    sigma = float(input("Enter the radius (sigma) for Gaussian kernel [default=2.0]: ") or "2.0")
    
    # Set output folder
    output_folder = os.path.join(os.path.dirname(input_file), "imagej_gaussian_output")
    
    # Process the file
    process_tiff_with_imagej_gaussian(input_file, output_folder, sigma)


if __name__ == "__main__":
    main()
