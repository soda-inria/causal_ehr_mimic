# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
import forestplot as fp

COHORT_NAME2LABEL = {
    "albumin_for_sepsis__obs_0f25d": "6h",
    "albumin_for_sepsis__obs_1d": "24h",
    "albumin_for_sepsis__obs_3d": "72h",
}

# %%
expe_name = "immortal_time_bias_double_robust_forest_agg_last"
results = pd.read_parquet(DIR2EXPERIENCES / expe_name)
# %%
# Create nice labels for forest plot
outcome_name = results["outcome_name"].unique()[0]
results["observation_period"] = results["cohort_name"].map(
    lambda x: COHORT_NAME2LABEL[x] if x in COHORT_NAME2LABEL.keys() else x
)
results["label"] = (
    "Agg="
    + results["event_aggregation"].map(
        lambda x: x.split(".")[-1].replace("()", "")
    )
    + ",\n  Est="
    + results["estimation_method"].map(
        lambda x: IDENTIFICATION2LABELS[x]
        if x in IDENTIFICATION2LABELS.keys()
        else x
    )
    + " + "
    + results["treatment_model"]
)
results.loc[
    results["estimation_method"] == "Difference in mean", "label"
] = IDENTIFICATION2LABELS["Difference in mean"]
results["estimation_method"] = results["estimation_method"].map(
    lambda x: IDENTIFICATION2LABELS[x]
    if x in IDENTIFICATION2LABELS.keys()
    else x
)
results["sortby"] = (
    results["treatment_model"] + "_" + results["event_aggregation"]
)

print(
    results.groupby(["estimation_method", "treatment_model", "cohort_name"])[
        "event_aggregation"
    ].count()
)
# %%
fp.forestplot(
    results,  # the dataframe with results data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel="label",  # column containing variable label
    ylabel="Confidence interval",  # y-label title
    xlabel=f"ATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    groupvar="observation_period",  # group variable
    group_order=list(COHORT_NAME2LABEL.values()),
    figsize=(5, 7),
    color_alt_rows=True,
    # sortby="sortby",
)
path2img = DIR2DOCS_IMG / expe_name
path2img.mkdir(exist_ok=True, parents=True)
plt.savefig(path2img / f"{expe_name}.pdf", bbox_inches="tight")
# %%
