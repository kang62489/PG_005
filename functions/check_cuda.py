# Standard library imports
import os
from pathlib import Path

# Third-party imports
import pynvml
from numba import cuda
from rich.console import Console

console = Console()


def check_cuda() -> bool:
    """Check CUDA availability and print diagnostic information"""
    try:
        # First check if CUDA is actually available through Numba
        if cuda.is_available():
            device = cuda.get_current_device()
            console.print(
                f"[green]CUDA is available. Using device: {device.name.decode('utf-8')}",
            )
            console.print(f"[green]Compute Capability: {device.compute_capability}")
            console.print(
                f"[green]Max threads per block: {device.MAX_THREADS_PER_BLOCK}",
            )

            return True

        console.print("[bold red]CUDA is not available through Numba. Checking why...")

        # Check NVIDIA driver
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            device_name = pynvml.nvmlDeviceGetName(handle)
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            console.print("[yellow]NVIDIA driver is installed")
            console.print(f"[yellow]GPU: {device_name}")
            console.print(f"[yellow]Driver Version: {driver_version}")
        except (
            pynvml.NVMLError_DriverNotLoaded,
            pynvml.NVMLError_Uninitialized,
            pynvml.NVMLError_LibraryNotFound,
        ) as e:
            console.print(
                f"[bold red]NVIDIA driver not found or not properly installed: {e!s}",
            )
            return False

        # Check CUDA installation
        base_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if not base_path.exists():
            console.print("[bold red]Could not find CUDA installation directory")
            return False

        # Look for all CUDA versions
        versions = [str(path) for path in base_path.glob("v*")]
        if not versions:
            console.print(f"[bold red]No CUDA versions found in {base_path}")
        else:
            # Sort versions in descending order
            versions.sort(reverse=True)
            cuda_path = versions[0]
            os.environ["CUDA_PATH"] = cuda_path

            # Also set other CUDA environment variables
            os.environ["CUDA_HOME"] = cuda_path
            os.environ["PATH"] = f"{Path(cuda_path) / 'bin'};{os.environ['PATH']}"
            os.environ["PATH"] = f"{Path(cuda_path) / 'libnvvp'};{os.environ['PATH']}"

            console.print(
                f"[yellow]Found CUDA installation at: {cuda_path} and set it to CUDA_PATH, CUDA_HOME, and PATH",
            )

            # Try to reinitialize CUDA
            if cuda.is_available():
                device = cuda.get_current_device()
                console.print(
                    f"[green]Successfully initialized CUDA with device: {device.name}",
                )
                return True
            console.print("[bold red]Found CUDA but failed to initialize")
            # Print current environment variables for debugging
            console.print("\n[yellow]Current CUDA environment variables:")
            for key in ["CUDA_PATH", "CUDA_HOME", "PATH"]:
                console.print(f"[yellow]{key}: {os.environ.get(key, 'Not set')}")
            return False

    except (KeyError, TypeError, RuntimeError) as e:
        console.print(f"[bold red]Error checking CUDA: {e!s}")
    return False
