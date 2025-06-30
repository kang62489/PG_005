# Third-party imports
import os
import sys
import warnings
from contextlib import contextmanager

import imagej
import numpy as np
import scyjava
from rich.console import Console

# Suppress Java warnings
os.environ["JAVA_OPTS"] = (
    '-Djavax.accessibility.assistive_technologies=" " --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --enable-native-access=ALL-UNNAMED'
)
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()


# Create a context manager to redirect stderr temporarily
@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    original_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = original_stderr


def imagej_gaussian(image_stack: np.ndarray, sigma: float = 16.0) -> np.ndarray:
    """Process a multipage TIFF file with ImageJ's Gaussian blur."""
    # Initialize ImageJ with specific options to reduce warnings
    console.print("[cyan]Initializing ImageJ...")

    # Set Java options to suppress warnings
    scyjava.config.add_options("-Djava.awt.headless=true")
    scyjava.config.add_options("--illegal-access=permit")
    scyjava.config.add_options("--add-opens=java.base/java.lang=ALL-UNNAMED")
    scyjava.config.add_options("--add-opens=java.base/java.util=ALL-UNNAMED")
    scyjava.config.add_options("--enable-native-access=ALL-UNNAMED")

    # Suppress Java stderr warnings
    with suppress_stderr():
        # Initialize with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ij = imagej.init("net.imagej:imagej", mode="headless")

        n_frames, height, width = image_stack.shape
        console.print(f"[bold green]Processing {n_frames} frames with Gaussian sigma={sigma}")

        # Process each frame
        gaussian_stack = np.zeros_like(image_stack)
        for frames_idx in range(n_frames):
            # Convert numpy array to ImageJ image
            ij_img = ij.py.to_java(image_stack[frames_idx])

            # Apply Gaussian blur using ImageJ
            # Get the op service and apply the filter
            blurred = ij.op().filter().gauss(ij_img, sigma)

            # Convert back to numpy array
            result_img = ij.py.from_java(blurred)
            gaussian_stack[frames_idx] = result_img

    # Clean up Java resources
    scyjava.shutdown_jvm()

    return gaussian_stack
