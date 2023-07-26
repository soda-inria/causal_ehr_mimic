# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from caumim.constants import DIR2DOCS_IMG, DIR2EXPERIENCES

# %%
dir2results = (
    DIR2EXPERIENCES
    / "predictive_failure__2023-07-21_14-58-34__obs_1/scores.csv"
)
dir2pretreatment_results = (
    DIR2EXPERIENCES
    / "predictive_failure__2023-07-25_22-04-47__obs_1__post_treatment_False/scores.csv"
)
post_treatment_results = pd.read_csv(dir2results)
pretreatment_results = pd.read_csv(dir2pretreatment_results)
# keep only the test set and rename

LABEL_MODEL_FEATURES = "Model features"
METRIC_LABELS = {
    "pr_auc": "PR AUC",
    "roc_auc": "ROC AUC",
}
pretreatment_results[LABEL_MODEL_FEATURES] = "Pre-treatment only"
post_treatment_results[LABEL_MODEL_FEATURES] = "All stay"

all_results = pd.concat([pretreatment_results, post_treatment_results])


def label_split(x):
    if "test" in x:
        return "Pre-treatment only"
    elif "val" in x:
        return "All stay"


path2img = DIR2DOCS_IMG / "predictive_failure"
path2img.mkdir(exist_ok=True, parents=True)
# %%
# single metric with both models
metric = "roc_auc"
results_metric = all_results.melt(
    id_vars=["random_seed", LABEL_MODEL_FEATURES],
    var_name="split",
    value_vars=[f"test_{metric}", f"val_{metric}"],
    value_name="score",
)
results_metric = results_metric[
    ~(
        results_metric["split"].str.contains("val_")
        & (results_metric[LABEL_MODEL_FEATURES] == "Pre-treatment only")
    )
]
results_metric["split"] = results_metric["split"].apply(label_split)
fig, ax = plt.subplots(figsize=(5, 2))
sns.boxplot(
    ax=ax,
    data=results_metric,
    y="split",
    x="score",
    hue=LABEL_MODEL_FEATURES,
    dodge=False,
)
ax.set_ylabel("Test set features")
# ax.get_legend().remove()
ax.set_xlabel(METRIC_LABELS[metric])
plt.savefig(path2img / f"{metric}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{metric}.jpg", bbox_inches="tight")

print(results_metric.groupby("split").mean())
# %% multiple metrics
metrics = ["pr_auc", "roc_auc"]
fig, axes = plt.subplots(1, len(metrics), figsize=(5, 2), sharey=True)
for i, (ax, metric) in enumerate(zip(axes, metrics)):
    results_metric = results.melt(
        id_vars=["random_seed"],
        var_name="split",
        value_vars=[f"test_{metric}", f"val_{metric}"],
        value_name="score",
    )
    results_metric["split"] = results_metric["split"].apply(label_split)
    sns.boxplot(
        ax=ax,
        data=results_metric,
        y="split",
        x="score",
        hue="split",
        dodge=False,
    )
    if i == 0:
        ax.set_ylabel("Features")
    else:
        ax.set_ylabel("")

    ax.get_legend().remove()
    ax.set_xlabel(METRIC_LABELS[metric])

# %%
