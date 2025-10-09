from pathlib import Path

from loguru import logger

# +----------------------------------------------------------+
# Path Configuration
# +----------------------------------------------------------+

PROJ_ROOT = Path(__file__).resolve().parents[2]  # Root directory

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
BACKUPS_DIR = DATA_DIR / "backups"

MODEL_DIR = PROJ_ROOT / "models"
LOG_DIR = PROJ_ROOT / "logs"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BENCHMARKS_DIR = REPORTS_DIR / "benchmarks"


logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
