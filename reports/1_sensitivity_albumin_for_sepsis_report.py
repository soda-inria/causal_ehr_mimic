# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
from caumim.reports_utils import add_rct_gold_standard_line

IS_MAIN_FIGURE = True
# %%
cohort_dir = create_cohort_folder(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
cohort_name = cohort_dir.name
expe_name = "estimates_20230523__est_lr_rf"  #
expe_name = "estimates_20230712__est_lr_rf__bs_50"
# expe_name = "estimates_20230516203739"
### For IP matching, interesting results with RF which seems to overfit the data and results are dependents on the aggregation strategy.
raw_results = pd.read_parquet(
    DIR2EXPERIENCES / cohort_name / expe_name / "logs"
)

if IS_MAIN_FIGURE:
    # mask the first aggregation which does not affect the results
    mask_forest = raw_results["estimation_method"] == "CausalForest"
    mask_last_agg = raw_results["event_aggregations"] == "['last']"
    # mask causal forests
    results = add_rct_gold_standard_line(
        raw_results[~mask_last_agg & ~mask_forest]
    )
else:
    results = add_rct_gold_standard_line(raw_results)
outcome_name = COLNAME_MORTALITY_28D

# %%
compute_times = results[
    ["estimation_method", "compute_time", "outcome_model", "event_aggregations"]
].loc[
    ~results["estimation_method"].isin(
        ["Difference in mean", LABEL_RCT_GOLD_STANDARD_ATE]
    )
]
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
    + results["treatment_model"]
)
results.loc[
    results["estimation_method"].isin(
        ["Difference in mean", LABEL_RCT_GOLD_STANDARD_ATE]
    ),
    "label",
] = ""
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
# %%
import forestplot as fp

# TODO: add the NTV on the forest plot.
axes = fp.forestplot(
    results,  # the dataframe with resultcodes data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel="label",  # column containing variable label
    ylabel="Confidence interval",  # y-label title
    xlabel=f"ATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    # annote=["treatment_model", "event_aggregation"],  # columns to annotate
    groupvar="estimation_method",  # group variable
    group_order=[
        ident_
        for ident_ in list(IDENTIFICATION2LABELS.values())
        if ident_ in results["estimation_method"].unique()
    ],
    # rightannote=["ntv"],  # columns to report on right of plot
    # right_annoteheaders=["Overlap measure as Normalized Total Variation"],
    figsize=(5, 12),
    color_alt_rows=True,
    sortby="sortby",
)
# axes.axvline(
#     VALUE_RCT_GOLD_STANDARD_ATE, linestyle="--", color="salmon", linewidth=2
# )
# axes.text(
#     VALUE_RCT_GOLD_STANDARD_ATE,
#     1,
#     LABEL_RCT_GOLD_STANDARD_ATE,
#     transform=axes.get_xaxis_transform(),
#     fontsize=12,
#     color="salmon",
# )
axes.set(xlim=(-0.15, 0.1))
path2img = DIR2DOCS_IMG / cohort_name
path2img.mkdir(exist_ok=True, parents=True)
if not IS_MAIN_FIGURE:
    expe_name = expe_name + "_supplementary"
plt.savefig(path2img / f"{expe_name}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{expe_name}.png", bbox_inches="tight")
compute_times.to_latex(path2img / f"compute_time_{expe_name}.tex")

# %%
