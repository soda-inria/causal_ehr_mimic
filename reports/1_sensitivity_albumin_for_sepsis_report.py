# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
from caumim.reports_utils import add_rct_gold_standard_line, add_albumin_label
from copy import deepcopy

IS_MAIN_FIGURE = True
SHARE_X_AXIS = True
# %%

cohort_dir = create_cohort_folder(deepcopy(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS))
cohort_name = cohort_dir.name
expe_name = "estimates_20230712__est_lr_rf__bs_50"
#    expe_name = "estimates_20230523__est_lr_rf__bs_10"
### For IP matching, interesting results with RF which seems to overfit the data and results are dependents on the aggregation strategy.
raw_results = pd.read_parquet(
    DIR2EXPERIENCES / cohort_name / expe_name / "logs"
)

if IS_MAIN_FIGURE:
    # mask the first aggregation which does not affect the results
    mask_causal_estimator = raw_results["estimation_method"].isin(
        ["CausalForest"]  # , "LinearDRLearner"]
    )
    mask_agg = raw_results["event_aggregations"].isin(
        ["['last']", "['first']"]
    )
    # mask causal forests
    results = add_rct_gold_standard_line(
        raw_results[~mask_agg & ~mask_causal_estimator]
    )
else:
    results = add_rct_gold_standard_line(raw_results)
outcome_name = COLNAME_MORTALITY_28D


mask_no_models = results["estimation_method"].isin(
    ["Difference in mean", LABEL_RCT_GOLD_STANDARD_ATE]
)
compute_times = results[
    [
        "estimation_method",
        "compute_time",
        "outcome_model",
        "event_aggregations",
    ]
].loc[~mask_no_models]
if IS_MAIN_FIGURE:
    results["label"] = "Est=" + results["treatment_model"].map(
        lambda x: ESTIMATORS2LABELS[x] if x in ESTIMATORS2LABELS.keys() else x
    )
else:
    results["label"] = (
        "Agg="
        + results["event_aggregations"].map(
            lambda x: x.split(".")[-1].replace("()", "")
        )
        # + "\nidentification:"
        # + results["estimation_method"].map(
        #     lambda x: IDENTIFICATION2LABEL[x]
        #     if x in IDENTIFICATION2LABEL.keys()
        #     else x
        # )
        + ", Est="
        + results["treatment_model"].map(
            lambda x: ESTIMATORS2LABELS[x]
            if x in ESTIMATORS2LABELS.keys()
            else x
        )
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
    results.groupby(["estimation_method", "treatment_model"])[
        "event_aggregations"
    ].count()
)
group_order = [NO_MODEL_GROUP_LABEL] + [
    ident_
    for ident_ in list(IDENTIFICATION2LABELS.values())
    if ident_ in results["estimation_method"].unique()
]
# %%
import forestplot as fp

if IS_MAIN_FIGURE:
    figsize = (3, 4.5)
else:
    figsize = (5, 12)
if SHARE_X_AXIS:
    figsize = (5, 4)

axes = fp.forestplot(
    results,  # the dataframe with resultcodes data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel="label",  # column containing variable label
    xlabel=f"ATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    # annote=["treatment_model", "event_aggregation"],  # columns to annotate
    groupvar="estimation_method",  # group variable
    group_order=group_order,
    figsize=figsize,
    color_alt_rows=True,
    sortby="sortby",
    ylabel="ATE (95% bootstrap confidence interval)",  # ylabel to print
    **{"marker": "D", "ylabel1_size": 10, "ylabel1_fontweight": "normal"},
)
axes.set(xlim=(-0.15, 0.1))

if IS_MAIN_FIGURE:
    outlier_x = 12
    x_less = 1.15
    fontsize = 7.5
else:
    outlier_x = 20
    x_less = 1.05
    fontsize = 10
if SHARE_X_AXIS:
    axes.set(xlim=SHARED_X_LIM)
    x_less = 1.0
axes.text(0.1, outlier_x, "Outlier ▶", ha="right", va="center")
axes = add_albumin_label(axes, x_less=x_less, fontsize=fontsize)

path2img = DIR2DOCS_IMG / cohort_name

path2img.mkdir(exist_ok=True, parents=True)
if not IS_MAIN_FIGURE:
    expe_name = expe_name + "_supplementary"
if SHARE_X_AXIS:
    expe_name = expe_name + "_shared_x_axis"
plt.savefig(path2img / f"{expe_name}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{expe_name}.png", bbox_inches="tight", dpi=300)
compute_times.to_latex(path2img / f"compute_time_{expe_name}.tex")

# %%
