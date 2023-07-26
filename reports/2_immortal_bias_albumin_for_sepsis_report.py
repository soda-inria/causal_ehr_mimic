# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
import forestplot as fp

from caumim.reports_utils import (
    add_rct_gold_standard_line,
    compute_gold_standard_ate_caironi,
)

COHORT_NAME2LABEL = {
    "albumin_for_sepsis__obs_0f25d": "6h",
    "albumin_for_sepsis__obs_1d": "24h",
    "albumin_for_sepsis__obs_3d": "72h",
}

# %%
# expe_name = "immortal_time_bias_double_robust_forest_agg_last__bs_50"
expe_name = "immortal_time_bias_double_robust_forest_agg_first_last__bs_30"
# expe_name = "immortal_time_bias_double_robust_forest_agg_first_last"
results = pd.read_parquet(DIR2EXPERIENCES / expe_name / "logs")
# results = add_rct_gold_standard_line(results)
# Create nice labels for forest plot
mask_no_models = results["estimation_method"].isin(
    ["Difference in mean", LABEL_RCT_GOLD_STANDARD_ATE]
)
outcome_name = results["outcome_name"].unique()[0]
results["observation_period"] = results["cohort_name"].map(
    lambda x: COHORT_NAME2LABEL[x] if x in COHORT_NAME2LABEL.keys() else x
)
# drop models with no models
results = results.loc[~mask_no_models]
results["label"] = (
    "Agg="
    + results["event_aggregations"].map(
        lambda x: x.split(".")[-1].replace("()", "")
    )
    + ", Est="
    + results["estimation_method"].map(
        lambda x: IDENTIFICATION2SHORT_LABELS[x]
        if x in IDENTIFICATION2SHORT_LABELS.keys()
        else x
    )
    + " + "
    + results["treatment_model"]
)
results.loc[mask_no_models, "label"] = results.loc[
    mask_no_models, "estimation_method"
]
NO_MODEL_GROUP_LABEL = ""
results.loc[mask_no_models, "estimation_method"] = NO_MODEL_GROUP_LABEL
results["estimation_method"] = results["estimation_method"].map(
    lambda x: IDENTIFICATION2LABELS[x]
    if x in IDENTIFICATION2LABELS.keys()
    else x
)
results["sortby"] = (
    results["treatment_model"] + "_" + results["event_aggregations"]
)

print(
    results.groupby(["estimation_method", "treatment_model", "cohort_name"])[
        "event_aggregations"
    ].count()
)
# %%
axes = fp.forestplot(
    results,  # the dataframe with results data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel="label",  # column containing variable label
    xlabel=f"ATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    groupvar="observation_period",  # group variable
    group_order=list(COHORT_NAME2LABEL.values()),
    figsize=(4, 4),
    color_alt_rows=True,
    ylabel="ATE (95% bootstrap confidence interval)",  # ylabel to print
    **{"marker": "D", "ylabel1_size": 10, "ylabel1_fontweight": "normal"},
)
axes.set(xlim=(-0.075, 0.075))

path2img = DIR2DOCS_IMG / expe_name
path2img.mkdir(exist_ok=True, parents=True)
# plt.savefig(path2img / f"{expe_name}.pdf", bbox_inches="tight")
# plt.savefig(path2img / f"{expe_name}.png", bbox_inches="tight")
# %%
