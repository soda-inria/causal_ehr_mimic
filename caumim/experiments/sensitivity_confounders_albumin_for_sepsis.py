from sklearn.utils import Bunch
import polars as pl

from caumim.constants import (
    COLNAME_EMERGENCY_ADMISSION,
    COLNAME_INSURANCE_MEDICARE,
    COLNAME_MORTALITY_28D,
    COLNAME_VALUE,
    DIR2COHORT,
)
from caumim.experiments.configurations import ESTIMATOR_RF, ESTIMATOR_RIDGE
from caumim.experiments.sensitivity_albumin_for_sepsis import (
    run_sensitivity_experiment,
)
from caumim.variables.selection import (
    LABEL_ALL_FEATURES,
    LABEL_DEMOGRAPHICS,
    LABEL_WO_DRUGS,
    LABEL_WO_MEASUREMENTS,
)


confounders_config = Bunch(
    **{
        "outcome_name": COLNAME_MORTALITY_28D,
        "experience_grid_dict": {
            "event_aggregations": [
                {
                    "first": pl.col(COLNAME_VALUE).first(),
                    "last": pl.col(COLNAME_VALUE).last(),
                },
            ],
            "estimation_method": [
                "DML",
                # "DRLearner",
                # "backdoor.propensity_score_weighting",
            ],
            "estimator": [
                # ESTIMATOR_RIDGE,
                ESTIMATOR_RF
            ],
            "feature_subset": [
                LABEL_WO_DRUGS,
                LABEL_DEMOGRAPHICS,
                LABEL_WO_MEASUREMENTS,
                LABEL_ALL_FEATURES,
            ],
        },
        "fraction": 1,
        "random_state": 0,
        "bootstrap_num_samples": 50,
    }
)

cohort_names = [
    "albumin_for_sepsis__obs_1d",
]

if __name__ == "__main__":
    for cohort_name_ in cohort_names:
        cohort_config = Bunch(**confounders_config.copy())
        cohort_config["cohort_folder"] = DIR2COHORT / cohort_name_
        cohort_config[
            "expe_name"
        ] = "sensitivity_confounders_albumin_for_sepsis"
        run_sensitivity_experiment(cohort_config)
