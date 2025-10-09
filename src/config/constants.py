import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from .env import get_env_variable

# +---------------------------------------------+
# Environment variables
# +---------------------------------------------+

# Load environment-specific .env file
env = os.getenv("ENVIRONMENT", "dev")
dotenv_path = find_dotenv(f".env.{env}", usecwd=False)
dotenv_path = Path(dotenv_path)

if loaded := load_dotenv(dotenv_path, verbose=True):
    logger.info(f"Env variables loaded from {dotenv_path.name}")
else:
    logger.error("Env variables not found")
    sys.exit(1)


# +----------------------------------------------------------+
# Experiment configuration
# +----------------------------------------------------------+

NIXTLA_API_KEY = get_env_variable("NIXTLA_API_KEY", required=False)

API_URL = "https://api.deelfietsdashboard.nl/dashboard-api/public/vehicles_in_public_space"
REFRESH_INTERVAL = 60
MAX_RETRIES = 5

DATE_FORMAT = "%Y%m%d_%H%M%S"
CHUNK_SIZE = 10**6

# +----------------------------------------------------------+
# Default parameters
# +----------------------------------------------------------+
DATASET = "rotterdam"

match DATASET:
    case "shared_mob" | "rotterdam" | "amsterdam" | "hague":
        TEMPORAL_COLUMNS = ["crowd"]
        TIME_COLUMN = "timestamp"
        GROUP_COLUMN = "district_id"
    case "citi" | "divvy":
        TEMPORAL_COLUMNS = ["arrivals", "departures"]
        TIME_COLUMN = "timestamp"
        GROUP_COLUMN = "station_id"
    case _:
        TEMPORAL_COLUMNS = ["y"]
        TIME_COLUMN = "ds"
        GROUP_COLUMN = "unique_id"


TARGET_COLUMN = [f"target_{col}" for col in TEMPORAL_COLUMNS]


TARGET = "Rotterdam Centrum"
CITY = "Rotterdam"

N_SPLITS = 5
FH = 60

DEFAULT_LAGS = [1, 5, 10, 15]
DEFAULT_WINDOWS = [5, 10, 15]
RANDOM_SEED = 42
LOG_INTERVAL = 10

INVALID_DISTRICTS = {
    "Rotterdam": {
        "Bedrijventerrein Schieveen",
        "Botlek",
        "Hoek van Holland",
        "Hoogvliet",
        "Pernis",
        "Rivium",
        "Vondelingenplaat",
        "Waalhaven",
        "Rozenburg",
    },
    "Amsterdam": {
        "Driemond",
        "Kadoelen",
        "Nellestein",
        "Nieuwendammerdijk/Buiksloterdijk",
        "Spieringhorn",
        "Tuindorp Buiksloot",
        "Tuindorp Nieuwendam",
        "Waterland",
    },
    "Hague": {
        "Kraayenstein en de Uithof",
    },
}
