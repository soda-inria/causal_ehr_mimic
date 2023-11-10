# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.reports_utils import add_albumin_label, add_rct_gold_standard_line
from caumim.variables.selection import FEATURE_SETS, LABEL_ALL_FEATURES

# %%
MAIN_FIGURE = True
SHARE_X_AXIS = False

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
# results = results.loc[~mask_no_models].reset_index(drop=True)

results["estimation_method"] = results["estimation_method"].map(
    lambda x: IDENTIFICATION2LABELS[x]
    if x in IDENTIFICATION2LABELS.keys()
    else x
)

mask_all_features = results["feature_subset"] == LABEL_ALL_FEATURES
mask_forest = results["treatment_model"] == "Forests"
mask_dml = results["estimation_method"] == "DML"
mask_unique_label = mask_all_features & mask_forest & mask_dml
mask_dr = results["estimation_method"] == "DRLearner"

if MAIN_FIGURE:
    results = results.loc[(mask_forest & mask_dr) | mask_no_models]
else:
    # Forest DML and DR have exactly the same
    # results for two feature sets, change the label for DML line:
    results.loc[mask_unique_label, "treatment_model"] = results.loc[
        mask_unique_label, "treatment_model"
    ].map(lambda x: x + ".")

results["feature_subset_clean"] = results["feature_subset"].map(
    lambda x: x + f" ({len(FEATURE_SETS[x])} features)"
    if x in FEATURE_SETS.keys()
    else x
)
results["label"] = (
    "Est="
    + results["estimation_method"]
    + " + "
    + results["treatment_model"].map(
        lambda x: ESTIMATORS2LABELS_SHORT[x]
        if x in ESTIMATORS2LABELS.keys()
        else x
    )
)
if MAIN_FIGURE:
    results["label"] = results["feature_subset_clean"]

results.loc[mask_no_models, "label"] = results.loc[
    mask_no_models, "estimation_method"
]

NO_MODEL_GROUP_LABEL = ""
results.loc[mask_no_models, "estimation_method"] = NO_MODEL_GROUP_LABEL
results.loc[mask_no_models, "feature_subset_clean"] = NO_MODEL_GROUP_LABEL

results["estimation_method"] = results["estimation_method"].map(
    lambda x: IDENTIFICATION2LABELS[x]
    if x in IDENTIFICATION2LABELS.keys()
    else x
)

if MAIN_FIGURE:

    def sort_feature_subset(feature_subset, label):
        if feature_subset in FEATURE_SETS.keys():
            return len(FEATURE_SETS[feature_subset])
        elif label == "Unajusted risk difference":
            return -1000
        else:
            # case rct
            return 1000

    results["sortby"] = results.apply(
        lambda x: sort_feature_subset(x["feature_subset"], x["label"]), axis=1
    )
else:
    results["sortby"] = (
        results["feature_subset_clean"]
        + "_"
        + results["treatment_model"]
        + "_"
        + results["estimation_method"]
    )
#
print(
    results.groupby(
        ["feature_subset_clean", "estimation_method", "treatment_model"]
    )["event_aggregations"].count()
)
results["ntv"] = results["ntv"].map(lambda x: f"{x:.2f}" if x > 0 else "")
# group_order = [NO_MODEL_GROUP_LABEL] + [ ident_ for ident_ in
#     list(IDENTIFICATION2LABELS.values()) if ident_ in
#     results["estimation_method"].unique() ]
#


group_order = [
    NO_MODEL_GROUP_LABEL,
    "All confounders (24 features)",
    "Without drugs (19 features)",
    "Without measurements (13 features)",
    "Socio-demographics (5 features)",
]
results = results.sort_values(by="sortby").reset_index(drop=True)
# results["result_ate_clean"] = results.apply(
#     lambda x: f"{x[RESULT_ATE]:.2f} ({x[RESULT_ATE_LB]:.2f}, {x[RESULT_ATE_UB]:.2f})",
#     axis=1,
# )

# %%
import forestplot as fp

if MAIN_FIGURE:
    group_order = None
    group_var = None
    figsize = (4, 4)
    rightannote = None
    right_annoteheaders = None
    x_less = 1.1
    fontsize = 9
    y_albumin = 1.05
else:
    group_var = "feature_subset_clean"
    figsize = (4, 12)
    rightannote = ["ntv"]
    right_annoteheaders = ["Overlap \n (NTV)"]
    x_less = 1.32
    fontsize = 10
    y_albumin = 0.95
if SHARE_X_AXIS:
    figsize = (5, 3)
axes = fp.forestplot(
    results,  # the dataframe with resultcodes data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel="label",  # column containing variable label
    xlabel=f"ATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    groupvar=group_var,  # group variable
    group_order=group_order,
    rightannote=rightannote,  # columns to report on right of plot
    right_annoteheaders=right_annoteheaders,
    figsize=figsize,
    color_alt_rows=True,
    sortby="sortby",
    ylabel="ATE (95% bootstrap confidence interval)",  # ylabel to print
    **{"marker": "D", "ylabel1_size": 10, "ylabel1_fontweight": "normal"},
)
axes = add_albumin_label(axes, x_less=x_less, fontsize=fontsize, y=y_albumin)

if MAIN_FIGURE:
    figname = cohort_name + "_main_figure"
    axes.set_xlim((-0.08, 0.05))
else:
    figname = cohort_name
if SHARE_X_AXIS:
    axes.set(xlim=SHARED_X_LIM)
if SHARE_X_AXIS:
    figname = figname + "_shared_x_axis"
path2img = DIR2DOCS_IMG / cohort_name
path2img.mkdir(exist_ok=True, parents=True)
plt.savefig(path2img / f"{figname}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{figname}.png", bbox_inches="tight", dpi=300)

# %%
