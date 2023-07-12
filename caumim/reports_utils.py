import pandas as pd
import numpy as np
from caumim.constants import (
    LABEL_RCT_GOLD_STANDARD_ATE,
    RESULT_ATE_LB,
    RESULT_ATE_UB,
    VALUE_RCT_GOLD_STANDARD_ATE,
    VALUE_RCT_GOLD_STANDARD_ATE_LB,
    VALUE_RCT_GOLD_STANDARD_ATE_UB,
)


def compute_gold_standard_ate_ci_binary(
    n_treatment: int,
    n_treatment_outcome: int,
    n_control: int,
    n_control_outcome: int,
):
    # Computed using https://training.cochrane.org/handbook/current/chapter-06#section-6-7-1
    mean_ate = n_treatment_outcome / n_treatment - n_control_outcome / n_control
    standard_error_ate = np.sqrt(
        n_treatment_outcome / n_treatment**2
        + n_control_outcome / n_control**2
    )
    # Not sure about normality assumption for a probability bounded between 0 and 1.
    lower_bound = mean_ate - 1.96 * standard_error_ate
    upper_bound = mean_ate + 1.96 * standard_error_ate

    return mean_ate, lower_bound, upper_bound


def compute_gold_standard_ate_caironi():
    # Using Caironi trial : https://www.nejm.org/doi/10.1056/NEJMoa1305727?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200www.ncbi.nlm.nih.gov
    return compute_gold_standard_ate_ci_binary(
        n_treatment=895,
        n_treatment_outcome=285,
        n_control=900,
        n_control_outcome=288,
    )


def add_rct_gold_standard_line(
    results: pd.DataFrame, gold_standard_rct: str = "caironi"
):
    if gold_standard_rct == "caironi":
        (
            gold_standard_ate,
            gold_standard_lb,
            gold_standard_ub,
        ) = compute_gold_standard_ate_caironi()
    else:
        raise ValueError(
            f"Only caironi is implemented, got {gold_standard_rct}."
        )
    results_w_gold_standard = pd.concat(
        [
            pd.DataFrame.from_dict(
                {
                    "estimation_method": [LABEL_RCT_GOLD_STANDARD_ATE],
                    "treatment_model": ["A"],
                    "outcome_model": ["A"],
                    "event_aggregations": ["None"],
                    "ntv": [-1],
                    "ATE": [gold_standard_ate],
                    RESULT_ATE_LB: [gold_standard_lb],
                    RESULT_ATE_UB: [gold_standard_ub],
                    "observation_period": "24h",
                },
            ),
            results,
        ]
    ).reset_index(drop=True)
    return results_w_gold_standard
