# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *

from caumim.reports_utils import hist_plot_binary_treatment_hte
%load_ext autoreload
%autoreload 2

# %%
cohort_name = "cate_estimates_20230716__bs_10"
### For IP matching, interesting results with RF which seems to overfit the data and results are dependents on the aggregation strategy.
results = pd.read_parquet(
    DIR2EXPERIENCES / "albumin_for_sepsis__obs_1d" / cohort_name / "logs"
)
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
results["ntv"] = results["ntv"].map(lambda x: f"{x:.2f}" if x > 0 else "")

# %%
run_results = results.iloc[0]
result_columns = [
    "X_cate__White",
    "X_cate__Female",
    "X_cate__admission_age",
    "cate_predictions",
    "cate_lb",
    "cate_ub",
]
cate_results = pd.DataFrame({k: run_results[k] for k in result_columns})

# %%  [markdown]
## Race CATE
# %%
fig, ax = hist_plot_binary_treatment_hte(
    cate_feature_name="X_cate__White",
    target_set="cate_predictions",
    cate_results=cate_results,
)

# %% [markdown]
# ## Sex CATE
# %%
fig, ax = hist_plot_binary_treatment_hte(
    cate_feature_name="X_cate__Female",
    target_set="cate_predictions",
    cate_results=cate_results,
)

# %% [markdown]
# ## Age Cate
# %%
cate_results_ = cate_results.copy()
cate_results_.sort_values("X_cate__admission_age", inplace=True)
plt.scatter(
    cate_results_["X_cate__admission_age"],
    cate_results_["cate_predictions"],
)
plt.fill_between(
    cate_results_["X_cate__admission_age"], cate_results_["cate_lb"], 
    cate_results_["cate_ub"], alpha=.4
    )
# %%
path2img = DIR2DOCS_IMG / cohort_name
path2img.mkdir(exist_ok=True, parents=True)
plt.savefig(path2img / f"{cohort_name}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{cohort_name}.png", bbox_inches="tight")

# %%
