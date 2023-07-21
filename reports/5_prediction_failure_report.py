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
results = pd.read_csv(dir2results)

METRIC_LABELS = {
    "pr_auc": "PR AUC",
    "roc_auc": "ROC AUC",
}


def label_split(x):
    if "test" in x:
        return "Pre-treatment only"
    elif "val" in x:
        return "All stay"


path2img = DIR2DOCS_IMG / "predictive_failure"
path2img.mkdir(exist_ok=True, parents=True)
# %%
# single metric
metric = "pr_auc"
fig, ax = plt.subplots(figsize=(5, 2))

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
ax.set_ylabel("Features")
ax.get_legend().remove()
ax.set_xlabel(METRIC_LABELS[metric])
plt.savefig(path2img / f"{metric}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{metric}.jpg", bbox_inches="tight")
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
