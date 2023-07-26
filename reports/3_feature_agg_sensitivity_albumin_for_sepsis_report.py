# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.reports_utils import add_rct_gold_standard_line

# %%
cohort_name = "sensitivity_feature_aggregation_albumin_for_sepsis__bs_50"
### For IP matching, interesting results with RF which seems to overfit the data and results are dependents on the aggregation strategy.
raw_results = pd.read_parquet(DIR2EXPERIENCES / cohort_name / "logs")
results = add_rct_gold_standard_line(raw_results)
outcome_name = COLNAME_MORTALITY_28D

mask_no_models = results["estimation_method"].isin(
    ["Difference in mean", LABEL_RCT_GOLD_STANDARD_ATE]
)


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
results["ntv"] = results["ntv"].map(lambda x: f"{x:.2f}" if x > 0 else "")
group_order = [NO_MODEL_GROUP_LABEL] + [
    ident_
    for ident_ in list(IDENTIFICATION2LABELS.values())
    if ident_ in results["estimation_method"].unique()
]
# %%
import forestplot as fp

fp.forestplot(
    results,  # the dataframe with resultcodes data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel="label",  # column containing variable label
    xlabel=f"ATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    # annote=["treatment_model", "event_aggregation"],  # columns to annotate
    groupvar="estimation_method",  # group variable
    group_order=group_order,
    rightannote=["ntv"],  # columns to report on right of plot
    right_annoteheaders=["Overlap \n (NTV)"],
    figsize=(5, 10),
    color_alt_rows=True,
    sortby="sortby",
    ylabel="ATE (95% bootstrap confidence interval)",  # ylabel to print
    **{"marker": "D", "ylabel1_size": 10, "ylabel1_fontweight": "normal"},
)
path2img = DIR2DOCS_IMG / cohort_name
path2img.mkdir(exist_ok=True, parents=True)
plt.savefig(path2img / f"{cohort_name}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{cohort_name}.png", bbox_inches="tight")

# %%
