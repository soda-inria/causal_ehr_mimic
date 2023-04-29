from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


def log_estimate(estimate: Dict, estimate_folder: str):
    estimate_folder_path = Path(estimate_folder)
    estimate_folder_path.mkdir(parents=True, exist_ok=True)

    estimate_ = {k: [v] for k, v in estimate.items()}
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    estimate_["time_stamp"] = [current_time]
    pd.DataFrame(estimate_).to_parquet(
        str(estimate_folder_path / f"{current_time}.parquet")
    )
