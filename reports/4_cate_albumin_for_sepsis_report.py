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
cohort_name = "cate_estimates_20230716__bs_10_w_intercept"
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
path2img = DIR2DOCS_IMG / cohort_name


result_columns = [
    "X_cate__White",
    "X_cate__Female",
    "X_cate__admission_age",
    "cate_predictions",
    "cate_lb",
    "cate_ub",
]
path2img.mkdir(exist_ok=True, parents=True)
# %%
models_final = ["StatsModelsLinearRegression", "Ridge", "RandomForestRegressor"]

# %% [markdown]
# ## Race CATE
# %%
for model_final in models_final:
    run_results = results.loc[results["model_final"] == model_final].iloc[0]
    cate_results = pd.DataFrame({k: run_results[k] for k in result_columns})
 
    cate_feature_name = "X_cate__White"
    fig, ax = hist_plot_binary_treatment_hte(
        cate_feature_name=cate_feature_name,
        target_set="cate_predictions",
        cate_results=cate_results,
    )
    estimation_args_str  = f"est__{run_results['estimation_method']}__nuisances__{run_results['treatment_model']}__final_{model_final}"
    plt.savefig(path2img / f"{cate_feature_name}__{estimation_args_str}.pdf", bbox_inches="tight")
    plt.show()
# %% [markdown]
# ## Sex CATE
# %%
for model_final in models_final:
    run_results = results.loc[results["model_final"] == model_final].iloc[0]
    cate_results = pd.DataFrame({k: run_results[k] for k in result_columns})
    cate_feature_name = "X_cate__Female"
    fig, ax = hist_plot_binary_treatment_hte(
        cate_feature_name=cate_feature_name,
        target_set="cate_predictions",
        cate_results=cate_results,
    )
    plt.savefig(path2img / f"{cate_feature_name}__{estimation_args_str}.pdf", bbox_inches="tight")
    plt.show()
# %% [markdown]
# ## Age CATE
# %%
for model_final in models_final:
    run_results = results.loc[results["model_final"] == model_final].iloc[0]
    cate_results = pd.DataFrame({k: run_results[k] for k in result_columns})
    cate_results.sort_values("X_cate__admission_age", inplace=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    cate_feature_name = "X_cate__admission_age"
    ax.scatter(
        cate_results[cate_feature_name],
        cate_results["cate_predictions"],
        c=COLORMAP[0]
    )
    ax.fill_between(
        cate_results["X_cate__admission_age"], cate_results["cate_lb"], 
        cate_results["cate_ub"], alpha=.4, color=COLORMAP[0]
        )
    ax.set_xlabel("Age")
    ax.set_ylabel(LABEL_CATE)
    plt.savefig(path2img / f"{cate_feature_name}__{estimation_args_str}.pdf", bbox_inches="tight")
    plt.show()
# %%