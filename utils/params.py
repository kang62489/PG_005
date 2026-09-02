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
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
RAW_TIFFS_DIR = BASE_DIR / "raw_tiffs"
RAW_ABFS_DIR = BASE_DIR / "raw_abfs"
PROC_TIFFS_DIR = BASE_DIR / "proc_tiffs"

# REC_* column ordering (rec_data.db)
@dataclass
class ColumnSorter:
    """Class to store REC_* table column groupings/ordering for rec_data.db."""

    CORE_COLUMNS: tuple[str, ...] = ("Filename", "Timestamp", "OBJ", "EXC", "EMI", "FRAMES", "SLICE", "AT", "SENSOR", "PAIRED_ABF")
    FLUIDIC_COLUMNS: tuple[str, ...] = (
        "PUFF",
        "PUFF_CONC",
        "PUFF_DURATION",
        "PUFF_COUNT",
        "PUFF_GAP",
        "PUFF_PRESSURE",
        "BATHED_IN",
        "BATHED_CONC",
        "FLOWED",
    )
    IMG_COND_COLUMNS: tuple[str, ...] = ("EXPO", "LEVEL", "FPS")
    EPHY_COND_COLUMNS: tuple[str, ...] = ("HOLD", "STIM")
    OTHER_COLUMNS: tuple[str, ...] = ("ANIMAL_ID", "FILTER", "LIGHT_ON", "PUMP")
    MEMO_COLUMNS: tuple[str, ...] = ("SPIKES", "NOTE")
    IGNORE_COLUMNS: tuple[str, ...] = ("CAM_TRIG_MODE", "EXT_STIM_TRIG", "LED_TRIG_MODE", "PUFF_TIP_POS", "PUFF_TIP_SIZE", "Ra")


# UI Sizes
@dataclass
class UISizes:
    """Class to store UI sizes."""

    MAIN_WINDOW_SIZE: tuple[int, int] = (420, 700)

    # Shell: Main Window
    LW_DOR_SHELL_WIDTH: int = 280
    BTN_SHELL_SIZE: tuple[int, int] = (90, 90)

    # Popout: Query by DOR / Log View
    POPWIN_DOR_DETAIL_SIZE: tuple[int, int] = (950, 350)
    POPWIN_LOG_VIEW_SIZE: tuple[int, int] = (1300, 700)
    POPWIN_DATA_SELECTOR_SIZE: tuple[int, int] = (1400, 700)
    POPWIN_IMG_PROC_SIZE: tuple[int, int] = (950, 800)
    POPWIN_ALS_CORRECT_SIZE: tuple[int, int] = (1300, 800)
    POPWIN_ALIGN_SPIKE_SIZE: tuple[int, int] = (1300, 800)

    # Dialog: Pick List
    TE_PICK_LIST_PREVIEW_HEIGHT: int = 350

    # Tab: DOR Query
    LW_DOR_WIDTH: int = 300

    TV_ANIMALS_HEIGHT: int = 100
    TV_INJECTIONS_HEIGHT: int = 120

    TE_DESCRIPTIONS_HEIGHT: int = 240
    TE_FINDINGS_HEIGHT: int = 60
    TE_LOG_CONTENTS_HEIGHT: int = 240
    TE_FOLDER_STRUCTURE_HEIGHT: int = 40

    # Tab: Data Selector
    TE_DIRS_HEIGHT: int = 60
    GB_PICK_LIST_WIDTH: int = 400

    # Tab: Image Processing
    TV_PICK_LIST_HEIGHT: int = 300
    BTN_START_PROCESSING_HEIGHT: int = 45

    # Tab: ALS Correction
    LW_GAUSS_TIFFS_WIDTH: int = 250
    GB_PROC_INFO_WIDTH: int = 400
    BTN_RUN_CORRECT_HEIGHT: int = 45


    # Tab: Spike-aligned Analysis
    TE_EXPORT_PATH_HEIGHT: int = 60
    BTN_RUN_ANALYSIS_HEIGHT: int = 45
    BTN_BROWSE_WIDTH: int = 60
