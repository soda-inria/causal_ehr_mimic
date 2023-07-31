# %%
import re
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
import forestplot as fp

from caumim.reports_utils import (
    add_albumin_label,
    add_rct_gold_standard_line,
    compute_gold_standard_ate_caironi,
    add_leading_zero,
)

COHORT_NAME2LABEL = {
    "albumin_for_sepsis__obs_0f25d": "6h",
    "albumin_for_sepsis__obs_1d": "24h",
    "albumin_for_sepsis__obs_3d": "72h",
}
IS_MAIN_FIGURE = False
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
results = results.loc[~mask_no_models]
results["observation_period"] = results["cohort_name"].map(
    lambda x: COHORT_NAME2LABEL[x] if x in COHORT_NAME2LABEL.keys() else x
)
outcome_name = results["outcome_name"].unique()[0]
if IS_MAIN_FIGURE:
    # mask the first aggregation which does not affect the results
    mask_causal_estimator = results["estimation_method"].isin(["DRLearner"])
    mask_stats_model = results["treatment_model"].isin(["Forests"])
    results = results[mask_causal_estimator & mask_stats_model]
    # results["Group"] = results["observation_period"]
    results["label"] = "Observation period: " + results["observation_period"]
    results["sortby"] = results["observation_period"].map(
        lambda x: -int(re.search("(\d\d*)h", x).group(1))
    )
else:
    # drop models with no models
    results.reset_index(drop=True, inplace=True)
    results["label"] = "Observation period: " + results["observation_period"]
    results["Group"] = (
        "Est="
        + results["estimation_method"].map(
            lambda x: IDENTIFICATION2SHORT_LABELS[x]
            if x in IDENTIFICATION2SHORT_LABELS.keys()
            else x
        )
        + " + "
        + results["treatment_model"].map(
            lambda x: ESTIMATORS2LABELS_SHORT[x]
            if x in ESTIMATORS2LABELS_SHORT.keys()
            else x
        )
    )
    results["sortby"] = results["Group"]
    # .map(
    #     lambda x: -int(re.search("(\d\d*)h", x).group(1))
    # )

print(
    results.groupby(["estimation_method", "treatment_model", "cohort_name"])[
        "event_aggregations"
    ].count()
)
# %%
if IS_MAIN_FIGURE:
    xlim = (-0.05, 0.05)
    figsize = (6.5, 2)
    group_order = None
    sortby = "sortby"
    group = None
    varlabel = "label"
else:
    xlim = (-0.12, 0.05)
    figsize = (7, 8)
    sortby = "sortby"
    varlabel = "Group"
    group = "label"
    group_order = [
        "Observation period: 6h",
        "Observation period: 24h",
        "Observation period: 72h",
    ]
axes = fp.forestplot(
    results,  # the dataframe with results data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel=varlabel,  # column containing variable label
    xlabel=f"ATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    groupvar=group,  # group variable
    group_order=group_order,
    figsize=figsize,
    color_alt_rows=True,
    sortby=sortby,
    ylabel="ATE (95% bootstrap confidence interval)",  # ylabel to print
    **{"marker": "D", "ylabel1_size": 10, "ylabel1_fontweight": "normal"},
)
axes.set(xlim=xlim)
if IS_MAIN_FIGURE:
    x_less = 1.1
    fontsize = 9
else:
    x_less = 1.05
    fontsize = 10
axes = add_albumin_label(axes, x_less=x_less, fontsize=fontsize)

path2img = DIR2DOCS_IMG / expe_name
path2img.mkdir(exist_ok=True, parents=True)
if IS_MAIN_FIGURE:
    sup_str = ""
else:
    sup_str = "_supp"
plt.savefig(path2img / f"{expe_name}{sup_str}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{expe_name}{sup_str}.png", bbox_inches="tight")
# %%
