# Third-party imports
import numpy as np
from numba import cuda
from rich.console import Console

console = Console()


def test_cuda() -> bool:
    """Quick test to verify GPU functionality"""
    try:
        # Small test array
        test_data = np.ones((1000,), dtype=np.float32)
        d_test = cuda.to_device(test_data)

        @cuda.jit
        def test_kernel(arr: np.ndarray) -> None:
            idx = cuda.grid(1)
            if idx < arr.size:
                arr[idx] = 2.0

        test_kernel[2, 512](d_test)
        cuda.synchronize()
        result = d_test.copy_to_host()

        if np.allclose(result, 2.0):
            console.print("[green]GPU test successful!")
            return True
    except Exception as e:
        console.print(f"[bold red]GPU test error: {e!s}")
        return False
    else:
        console.print("[bold red]GPU test failed!")
        return False
