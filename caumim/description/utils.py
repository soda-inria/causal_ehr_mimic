import pandas as pd

from caumim.constants import (
    COLNAME_DELTA_INCLUSION_INTIME,
    COLNAME_DELTA_INTERVENTION_INCLUSION,
    COLNAME_DELTA_INTIME_ADMISSION,
    COLNAME_DELTA_MORTALITY,
    COLNAME_INCLUSION_START,
    COLNAME_INTERVENTION_START,
)

COMMON_DELTAS = [
    COLNAME_DELTA_MORTALITY,
    COLNAME_DELTA_INTERVENTION_INCLUSION,
    COLNAME_DELTA_INCLUSION_INTIME,
    COLNAME_DELTA_INTIME_ADMISSION,
    "los_hospital",
    "los_icu",
]


def add_delta(
    target_trial_population: pd.DataFrame,
):
    """Generate time deltas for target trial population in day unit.

    Args:
        target_trial_population (pd.DataFrame): _description_
        unit (str, optional): _description_. Defaults to "hours".
    Returns:
        _type_: _description_
    """
    unit_multiplier = 3600 * 24
    target_trial_population_w_delta = target_trial_population.copy()

    target_trial_population[COLNAME_DELTA_MORTALITY] = (
        target_trial_population["dod"]
        - target_trial_population[COLNAME_INCLUSION_START]
    ).dt.total_seconds() / unit_multiplier
    target_trial_population[COLNAME_DELTA_INTERVENTION_INCLUSION] = (
        target_trial_population[COLNAME_INTERVENTION_START]
        - target_trial_population[COLNAME_INCLUSION_START]
    ).dt.total_seconds() / unit_multiplier
    target_trial_population[COLNAME_DELTA_INCLUSION_INTIME] = (
        target_trial_population[COLNAME_INCLUSION_START]
        - target_trial_population["icu_intime"]
    ).dt.total_seconds() / unit_multiplier
    target_trial_population[COLNAME_DELTA_INTIME_ADMISSION] = (
        target_trial_population["icu_intime"]
        - target_trial_population["admittime"]
    ).dt.total_seconds() / unit_multiplier

    return target_trial_population_w_delta


def describe_delta(target_trial_population, unit="hours"):
    target_trial_population_ = target_trial_population.copy()
    if unit == "hours":
        for delta_ in COMMON_DELTAS:
            target_trial_population_[delta_] = (
                target_trial_population_[delta_] * 24
            )
    elif unit == "days":
        pass
    else:
        raise ValueError(f"Unit {unit} not supported.")
    return target_trial_population_[COMMON_DELTAS].describe().T
