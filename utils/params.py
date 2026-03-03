## Modules
# Standard Library imports
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Third-party imports
from PySide6.QtCore import QSize, Qt

# Application Info
APP_NAME = "CD-ROM"
APP_VERSION = "beta-0.1.0"
APP_AUTHOR = "Kang"
APP_LAST_UPDATE = f"{datetime.now():%Y-%b-%d}"
APP_STATUS_MESSAGE = f"{APP_NAME} {APP_VERSION}, Author: {APP_AUTHOR}, Last Update: {APP_LAST_UPDATE}, Made in OIST"

# Directory Paths
BASE_DIR = Path(__file__).parent.parent
STYLES_DIR = BASE_DIR / "styles"
MODELS_DIR = BASE_DIR / "data"


# UI Sizes
@dataclass
class UISizes:
    """Class to store UI sizes."""

    MAIN_WINDOW_SIZE: tuple[int, int] = (1600, 850)
