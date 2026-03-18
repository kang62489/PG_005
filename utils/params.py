## Modules
# Standard Library imports
import datetime
from dataclasses import dataclass
from pathlib import Path

# Application Info
APP_NAME = "ACID"
APP_VERSION = "beta-0.1.0"
APP_AUTHOR = "Kang"
APP_LAST_UPDATE = f"{datetime.datetime.now(tz=datetime.UTC):%Y-%b-%d}"
APP_STATUS_MESSAGE = f"{APP_NAME} {APP_VERSION}, Author: {APP_AUTHOR}, Last Update: {APP_LAST_UPDATE}, Made in OIST"

# Directory Paths
BASE_DIR = Path(__file__).parent.parent
STYLES_DIR = BASE_DIR / "styles"
MODELS_DIR = BASE_DIR / "data"
EXP_DB_PATH = MODELS_DIR / "exp_info.db"
REC_DB_PATH = MODELS_DIR / "rec_data.db"


# UI Sizes
@dataclass
class UISizes:
    """Class to store UI sizes."""

    MAIN_WINDOW_SIZE: tuple[int, int] = (1600, 850)

    # Tab: DOR Query
    LW_DOR_WIDTH: int = 300

    TV_ANIMALS_HEIGHT: int = 100
    TV_INJECTIONS_HEIGHT: int = 120
