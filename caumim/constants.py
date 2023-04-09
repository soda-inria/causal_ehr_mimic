import os
from pathlib import Path

from dotenv import load_dotenv

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

load_dotenv()

ROOT_DIR = Path(
    os.getenv(
        "ROOT_DIR", Path(os.path.dirname(os.path.abspath(__file__))) / ".."
    )
)

# Default paths
# Data
DIR2DATA = ROOT_DIR / "data"
DIR2RESOURCES = DIR2DATA / "resources"
DIR2EXPERIENCES = DIR2DATA / "experiences"
DIR2RESULTS = DIR2DATA / "results"
DIR2MIMIC = DIR2DATA / "mimiciv_as_parquet"

# Docs
DIR2DOCS = ROOT_DIR / "docs/source"
DIR2DOCS_STATIC = DIR2DOCS / "_static"
DIR2DOCS_IMG = DIR2DOCS_STATIC / "img"
DIR2DOCS_COHORT = DIR2DOCS_IMG / "cohort"
DIR2DOCS_EXPERIENCES = DIR2DOCS_IMG / "experiences"
DIR2DOCS_COHORT.mkdir(parents=True, exist_ok=True)
DIR2DOCS_EXPERIENCES.mkdir(parents=True, exist_ok=True)
