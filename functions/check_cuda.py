# ============================================================================
# PART 1: Environment Setup (BEFORE numba import)
# ============================================================================

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from rich.console import Console

console = Console()

# NOTE: numba is NOT imported yet - this is critical!


def _setup_cuda_environment() -> tuple[bool, str]:
    """
    Detect and configure CUDA environment variables BEFORE importing numba.
    This function runs automatically when the module is imported.

    Returns:
        tuple: (success: bool, message: str)
    """
    # Check if we're on Windows
    if sys.platform != "win32":
        # For Linux/Mac, let numba auto-detect
        return True, "Non-Windows system - using default CUDA detection"

    # Base CUDA installation path (Windows)
    base_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")

    if not base_path.exists():
        return False, f"CUDA installation directory not found: {base_path}"

    # Find all installed CUDA versions
    versions = sorted([path for path in base_path.glob("v*")], reverse=True)

    if not versions:
        return False, f"No CUDA versions found in {base_path}"

    # Categorize CUDA versions by major version
    cuda_11_versions = []
    cuda_12_versions = []
    other_versions = []

    for version_path in versions:
        version_name = version_path.name  # e.g., "v11.8", "v12.0"
        try:
            # Extract major version number
            major_version = int(version_name.split(".")[0].replace("v", ""))
            if major_version == 11:
                cuda_11_versions.append(version_path)
            elif major_version == 12:
                cuda_12_versions.append(version_path)
            else:
                other_versions.append(version_path)
        except ValueError:
            other_versions.append(version_path)

    # Select preferred version: CUDA 11.x first (best numba compatibility)
    preferred_version = None
    if cuda_11_versions:
        preferred_version = cuda_11_versions[0]  # Already sorted, highest 11.x
    elif cuda_12_versions:
        preferred_version = cuda_12_versions[0]  # Fallback to 12.x
    elif other_versions:
        preferred_version = other_versions[0]  # Last resort
    else:
        preferred_version = versions[0]

    cuda_path = str(preferred_version)

    # Set CUDA environment variables
    os.environ["CUDA_PATH"] = cuda_path
    os.environ["CUDA_HOME"] = cuda_path

    # Set Numba-specific CUDA environment variables
    nvvm_path = Path(cuda_path) / "nvvm" / "bin"
    nvvm_dll = None
    # Look for nvvm DLL (version number varies)
    for dll in nvvm_path.glob("nvvm64_*.dll"):
        nvvm_dll = str(dll)
        break

    if nvvm_dll:
        os.environ["NUMBAPRO_NVVM"] = nvvm_dll

    libdevice_path = Path(cuda_path) / "nvvm" / "libdevice"
    if libdevice_path.exists():
        os.environ["NUMBAPRO_LIBDEVICE"] = str(libdevice_path)

    # Set CUDA driver path
    nvcuda_dll = Path(r"C:\Windows\System32\nvcuda.dll")
    if nvcuda_dll.exists():
        os.environ["NUMBA_CUDA_DRIVER"] = str(nvcuda_dll)

    # Update PATH to prioritize selected CUDA version
    cuda_bin = Path(cuda_path) / "bin"
    cuda_libnvvp = Path(cuda_path) / "libnvvp"

    current_path = os.environ.get("PATH", "")

    # Remove other CUDA versions from PATH to avoid conflicts
    path_parts = current_path.split(";")
    cleaned_path = []
    for part in path_parts:
        # Skip paths containing other CUDA versions
        if "CUDA\\v" in part and cuda_path not in part:
            continue
        cleaned_path.append(part)

    # Add selected CUDA paths at the beginning
    new_path_parts = [str(cuda_bin), str(cuda_libnvvp)] + cleaned_path
    os.environ["PATH"] = ";".join(new_path_parts)

    version_name = preferred_version.name
    return True, f"CUDA environment configured for {version_name} at {cuda_path}"


# Run setup when module is imported (BEFORE numba import)
_SETUP_SUCCESS, _SETUP_MESSAGE = _setup_cuda_environment()


# ============================================================================
# PART 2: Import numba (AFTER environment setup)
# ============================================================================

from numba import cuda

# ============================================================================
# PART 3: CUDA checking function
# ============================================================================


def check_cuda() -> bool:
    """Check CUDA availability and print diagnostic information"""
    # Print setup status
    if _SETUP_SUCCESS:
        console.print(f"[dim]{_SETUP_MESSAGE}[/dim]")
    else:
        console.print(f"[yellow]Setup warning: {_SETUP_MESSAGE}[/yellow]")

    try:
        # Check if CUDA is available through Numba
        if cuda.is_available():
            device = cuda.get_current_device()
            # Handle both bytes and str for device name (compatibility with different numba versions)
            device_name = device.name.decode("utf-8") if isinstance(device.name, bytes) else device.name
            console.print(f"[green]CUDA is available. Using device: {device_name}[/green]")
            console.print(f"[green]Compute Capability: {device.compute_capability}[/green]")
            console.print(f"[green]Max threads per block: {device.MAX_THREADS_PER_BLOCK}[/green]")
            return True

        console.print("[bold red]CUDA is not available through Numba. Checking why...[/bold red]")

        # Check NVIDIA driver (only if pynvml is available)
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                device_name = pynvml.nvmlDeviceGetName(handle)
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                console.print("[yellow]NVIDIA driver is installed[/yellow]")
                console.print(f"[yellow]GPU: {device_name}[/yellow]")
                console.print(f"[yellow]Driver Version: {driver_version}[/yellow]")
                pynvml.nvmlShutdown()
            except (
                pynvml.NVMLError_DriverNotLoaded,
                pynvml.NVMLError_Uninitialized,
                pynvml.NVMLError_LibraryNotFound,
            ) as e:
                console.print(f"[bold red]NVIDIA driver not found or not properly installed: {e!s}[/bold red]")
                return False
        else:
            console.print("[yellow]pynvml not available - skipping detailed driver check[/yellow]")

        # Print current environment variables for debugging
        console.print("\n[yellow]Current CUDA environment variables:[/yellow]")
        for key in ["CUDA_PATH", "CUDA_HOME", "NUMBAPRO_NVVM", "NUMBAPRO_LIBDEVICE"]:
            value = os.environ.get(key, "Not set")
            # Truncate long paths for readability
            if len(value) > 100:
                value = value[:100] + "..."
            console.print(f"[yellow]{key}: {value}[/yellow]")

        return False

    except (KeyError, TypeError, RuntimeError) as e:
        console.print(f"[bold red]Error checking CUDA: {e!s}[/bold red]")
        return False
