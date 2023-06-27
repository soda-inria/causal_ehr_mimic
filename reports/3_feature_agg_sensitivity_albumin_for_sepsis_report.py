# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *

# %%
cohort_name = "sensitivity_feature_aggregation_albumin_for_sepsis"
### For IP matching, interesting results with RF which seems to overfit the data and results are dependents on the aggregation strategy.
results = pd.read_parquet(DIR2EXPERIENCES / cohort_name / "result_logs")
outcome_name = COLNAME_MORTALITY_28D

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
results.loc[results["estimation_method"] == "Difference in mean", "label"] = ""
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

fp.forestplot(
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
        id_label
        for id_label in list(IDENTIFICATION2LABELS.values())
        if id_label in results["estimation_method"].unique()
    ],
    rightannote=["ntv"],  # columns to report on right of plot
    right_annoteheaders=["Overlap measure as Normalized Total Variation"],
    figsize=(5, 12),
    color_alt_rows=True,
    sortby="sortby",
)
path2img = DIR2DOCS_IMG / cohort_name
path2img.mkdir(exist_ok=True, parents=True)
plt.savefig(path2img / f"{cohort_name}.pdf", bbox_inches="tight")
# %%
