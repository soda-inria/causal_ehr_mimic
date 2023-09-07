# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.reports_utils import add_albumin_label, add_rct_gold_standard_line
from caumim.variables.selection import FEATURE_SETS

# %%
cohort_name = "sensitivity_confounders_albumin_for_sepsis__bs_50"
### For IP matching, interesting results with RF which seems to overfit the data and results are dependents on the aggregation strategy.
raw_results = pd.read_parquet(DIR2EXPERIENCES / cohort_name / "logs")
results = add_rct_gold_standard_line(raw_results)
mask_dm = results["estimation_method"] == "Difference in mean"
results.loc[mask_dm, "treatment_model"] = "A"
results.loc[mask_dm, "outcome_model"] = "A"

results = results.loc[
    results["estimation_method"] != "backdoor.propensity_score_weighting"
].reset_index(drop=True)

outcome_name = COLNAME_MORTALITY_28D

mask_no_models = results["estimation_method"].isin(
    ["Difference in mean", LABEL_RCT_GOLD_STANDARD_ATE]
)
results = results.loc[~mask_no_models].reset_index(drop=True)

# %%
results["estimation_method"] = results["estimation_method"].map(
    lambda x: IDENTIFICATION2LABELS[x]
    if x in IDENTIFICATION2LABELS.keys()
    else x
)

results["label"] = (
    # "Features="
    # + results["feature_subset"]
    "Est="
    + results["estimation_method"]
    + " + "
    + +results["treatment_model"].map(
        lambda x: ESTIMATORS2LABELS_SHORT[x]
        if x in ESTIMATORS2LABELS.keys()
        else x
    )
)
results.loc[mask_no_models, "label"] = results.loc[
    mask_no_models, "estimation_method"
]
# NO_MODEL_GROUP_LABEL = ""
# results.loc[mask_no_models, "estimation_method"] = NO_MODEL_GROUP_LABEL
# results.loc[mask_no_models, "feature_subset"] = NO_MODEL_GROUP_LABEL
results["estimation_method"] = results["estimation_method"].map(
    lambda x: IDENTIFICATION2LABELS[x]
    if x in IDENTIFICATION2LABELS.keys()
    else x
)
results["sortby"] = results["feature_subset"] + "_" + results["treatment_model"]
#
print(
    results.groupby(["feature_subset", "estimation_method", "treatment_model"])[
        "event_aggregations"
    ].count()
)
results["ntv"] = results["ntv"].map(lambda x: f"{x:.2f}" if x > 0 else "")
# group_order = [NO_MODEL_GROUP_LABEL] + [
#     ident_
#     for ident_ in list(IDENTIFICATION2LABELS.values())
#     if ident_ in results["estimation_method"].unique()
# ]
results["feature_subset"] = results["feature_subset"].map(
    lambda x: x + f" ({len(FEATURE_SETS[x])} features)"
    if x in FEATURE_SETS.keys()
    else x
)
group_order = [
    "All confounders (24 features)",
    "Without drugs (19 features)",
    "Without measurements (13 features)",
    "Socio-demographics (5 features)",
]
# results["feature_subset"].unique()
# results = results.sort_values(by="sortby").reset_index(drop=True)
# %%
import forestplot as fp

axes = fp.forestplot(
    results,  # the dataframe with resultcodes data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel="label",  # column containing variable label
    xlabel=f"ATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    # annote=["treatment_model", "event_aggregation"],  # columns to annotate
    groupvar="feature_subset",  # group variable
    group_order=group_order,
    rightannote=["ntv"],  # columns to report on right of plot
    right_annoteheaders=["Overlap \n (NTV)"],
    figsize=(4, 12),
    color_alt_rows=True,
    sortby="sortby",
    ylabel="ATE (95% bootstrap confidence interval)",  # ylabel to print
    **{"marker": "D", "ylabel1_size": 10, "ylabel1_fontweight": "normal"},
    table=False,
)
axes = add_albumin_label(axes, x_less=1.18, fontsize=10)

path2img = DIR2DOCS_IMG / cohort_name
path2img.mkdir(exist_ok=True, parents=True)
plt.savefig(path2img / f"{cohort_name}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{cohort_name}.png", bbox_inches="tight")

# %%
