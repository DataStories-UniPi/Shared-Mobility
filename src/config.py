import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()  # Load environment variables from .env file if it exists


PROJ_ROOT = Path(__file__).resolve().parents[1]  # Root directory

# Path configurations
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODEL_DIR = PROJ_ROOT / "models"
LOG_DIR = PROJ_ROOT / "logs"
BENCHMARKS_DIR = PROJ_ROOT / "benchmarks"


# Experiment Consistency parameters
TARGET = "Rotterdam Centrum"  # Target district
CITY = "Rotterdam"  # Target city
N_SPLITS = 5  # Number of cross-validation splits
FH = 15  # Forecast horizon


logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
