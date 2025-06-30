# Third-party imports
import psutil


def get_memory_usage() -> str:
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024
