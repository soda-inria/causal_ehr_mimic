# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
import forestplot as fp

from caumim.reports_utils import hist_plot_binary_treatment_hte, add_albumin_label
%load_ext autoreload
%autoreload 2

# %%
cohort_name = "cate_estimates_20230718_w_septic_shock__bs_10"
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

cate_features = [
    "X_cate__septic_shock",
    "X_cate__admission_age",
    "X_cate__White",
    "X_cate__Female",
    ]
result_columns = [
    *cate_features,
    "cate_predictions",
    "cate_lb",
    "cate_ub",
]
path2img.mkdir(exist_ok=True, parents=True)
# %% [markdown] 
# CATE box plot
# %%
for model_final in ["Ridge", "RandomForestRegressor"]:
    run_results = results.loc[results["model_final"] == model_final].iloc[0]
    cate_results = pd.DataFrame({k: run_results[k] for k in result_columns})
    # binerize age to be consistent with other variables
    age_group_thres = 60
    cate_results["X_cate__admission_age_bin"] = cate_results["X_cate__admission_age"].apply(lambda x: x>=age_group_thres)
    # boxplot preprocesing
    results_boxplot = cate_results.melt(id_vars="cate_predictions", value_vars=[ "X_cate__admission_age_bin","X_cate__septic_shock", "X_cate__Female","X_cate__White"])
    results_boxplot["CATE feature"] = results_boxplot["variable"].apply(lambda x: LABEL_MAPPING_HTE_FEATURE_COL[x])
    results_boxplot["Group"] = results_boxplot.apply(lambda x: LABEL_MAPPING_HTE_BINARY_NAME[x["CATE feature"]][x["value"]], axis=1)


    fig, axes = plt.subplots(4, 1, figsize=(6, 3), sharex=True)
    for i, cate_feature in enumerate(results_boxplot["CATE feature"].unique()):
        ax = axes[i]
        plot_data_ = results_boxplot.loc[
            results_boxplot["CATE feature"] == cate_feature
        ]
        palette = {k:COLORMAP_HTE[k] for k in plot_data_["Group"].unique()}
        sns.boxplot(
            ax=ax,
            data=plot_data_, x="cate_predictions", 
            y="Group",
            #y="CATE feature",
            hue="Group",
            dodge=False,
            palette=palette,
            #kind="box", row="CATE feature",
            #height=5, aspect=1, 
            width=0.8
        )
        ax.get_legend().remove()
        if i !=3:
            ax.set(xlabel="")
        else:
            ax.set(xlabel="Distribution of Individual Treatment Effect")
        ax.set(ylabel=cate_feature)
    add_albumin_label(axes[0], x_less=1, x_more=0, y=1.15, fontsize=10)
    estimation_args_str  = f"est__{run_results['estimation_method']}__nuisances__{run_results['treatment_model']}__final_{model_final}"
    plt.savefig(path2img / f"boxplot_{estimation_args_str}.pdf", bbox_inches="tight")
    plt.savefig(path2img / f"boxplot_{estimation_args_str}.jpg", bbox_inches="tight")    
# %% [markdown]
# # Failure mode of the final forest model for age 
cate_feature_name = "X_cate__admission_age"

models_final = ["Ridge", "RandomForestRegressor"]
fix, ax = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
cat_white  = 1
cat_female = 0
cat_shock = 0
for i, model_final in enumerate(models_final):
    run_results = results.loc[results["model_final"] == model_final].iloc[0]
    cate_results = pd.DataFrame({k: run_results[k] for k in result_columns})
    cate_results_plot = cate_results.loc[
        (cate_results["X_cate__White"] == cat_white)
        &
         (cate_results["X_cate__Female"] == cat_female)
         & (cate_results["X_cate__septic_shock"] == cat_shock)
    ]
    sns.scatterplot(
        ax=ax[i],
        data=cate_results_plot,
        x=cate_feature_name,
        y="cate_predictions",
    )
    ax[i].set_ylabel(f"CATE with final\n {model_final}")
    ax[1].set_ylabel(f"CATE with final\n Random Forest")
ax[1].set_xlabel("Age at admission")
estimation_args_str  = f"est__{run_results['estimation_method']}__nuisances__{run_results['treatment_model']}__final_{model_final}" 
xp_name = f"cate_age_forest_failure_w{cat_white}_f{cat_female}_shock{cat_shock}_{estimation_args_str}"
plt.savefig(path2img / f"{xp_name}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"{xp_name}.jpg", bbox_inches="tight")
plt.show()
#%%
# # Failure mode of the final forest model for age 
cate_feature_name = "X_cate__admission_age"
def build_category(x):
    category_str = ""
    if x["X_cate__White"] == 1:
        category_str += "White "
    else:
        category_str += "Non white "
    if x["X_cate__Female"]:
        category_str += "female"
    else:
        category_str += "male"
    if x["X_cate__septic_shock"]:
        category_str += " w. septic shock"
    else:
        category_str += " wo. septic shock"
    return category_str
models_final = ["Ridge", "RandomForestRegressor"]
fix, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
for i, model_final in enumerate(models_final):
    run_results = results.loc[results["model_final"] == model_final].iloc[0]
    cate_results = pd.DataFrame({k: run_results[k] for k in result_columns})
    cate_results["Binary features"] = cate_results.apply(
        lambda x: build_category(x), axis=1
    )
    sns.scatterplot(
        ax=ax[i],
        data=cate_results,
        x=cate_feature_name,
        y="cate_predictions",
        alpha=0.5, 
        hue="Binary features",
        edgecolor='black',
        linewidth=0.4,
    )
    ax[i].set_ylabel(f"CATE with final\n {model_final}")
    handles, labels = ax[i].get_legend_handles_labels()
    ax[i].legend().remove()

ax[0].legend(handles, labels, title="Binary features", bbox_to_anchor=(1.05, 1), loc='upper left',
                      ncols=1, borderaxespad=0.,)
# plt.legend(
#     handles, labels, bbox_to_anchor=(1.05, 0.5),
# )
ax[1].set_ylabel(f"CATE with final\n Random Forest")
ax[1].set_xlabel("Age at admission")
estimation_args_str  = f"est__{run_results['estimation_method']}__nuisances__{run_results['treatment_model']}__final_{model_final}"
plt.savefig(path2img / f"cate_age_forest_failure_all_category__{estimation_args_str}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"cate_age_forest_failure_all_category__{estimation_args_str}.jpg", bbox_inches="tight")
plt.show()

# %% [markdown]
# # Cate distributions
# %% [markdown]
# ## Septic shock CATE
# %%
models_final = ["StatsModelsLinearRegression", "Ridge", "RandomForestRegressor"]
for cate_feature_name in ["X_cate__White","X_cate__Female","X_cate__admission_age_bin","X_cate__septic_shock"]:
    print(f"CATE analysis on {LABEL_MAPPING_HTE_FEATURE_COL[cate_feature_name]}")
    for model_final in models_final:
        run_results = results.loc[results["model_final"] == model_final].iloc[0]
        cate_results = pd.DataFrame({k: run_results[k] for k in result_columns})
        cate_results["X_cate__admission_age_bin"] = cate_results["X_cate__admission_age"].apply(lambda x: x>=age_group_thres)

        fig, ax = , add_albumin_label(
            cate_feature_name=cate_feature_name,
            target_set="cate_predictions",
            cate_results=cate_results,
        )
        estimation_args_str  = f"est__{run_results['estimation_method']}__nuisances__{run_results['treatment_model']}__final_{model_final}"
        plt.savefig(path2img / f"{cate_feature_name}__{estimation_args_str}.pdf", bbox_inches="tight")
        plt.savefig(path2img / f"{cate_feature_name}__{estimation_args_str}.jpg", bbox_inches="tight")
        plt.show()
# %% [markdown]
# ## Forest plot
# %%
# format the results for forest plot
cate_var_results_formatted_list = []
for cate_var_name in ["X_cate__White","X_cate__Female","X_cate__admission_age_bin","X_cate__septic_shock"]:
    cate_label = LABEL_MAPPING_HTE_FEATURE_COL[cate_var_name]
    label_mapping = LABEL_MAPPING_HTE_BINARY_NAME[cate_label]
    cate_var_results = cate_results.groupby(cate_var_name).agg(
        **{
            "CATE": pd.NamedAgg("cate_predictions", np.mean),
            "ub": pd.NamedAgg("cate_predictions", lambda x: np.quantile(x, 0.95)),
            "lb": pd.NamedAgg("cate_predictions", lambda x: np.quantile(x, 0.05))
            }
    ).reset_index()
    cate_var_results["Group"] = cate_var_results[cate_var_name].apply(lambda x: label_mapping[x])
    cate_var_results["CATE_feature"] = cate_label
    cate_var_results_formatted_list.append(cate_var_results.drop(columns=cate_var_name))
cate_results_formatted = pd.concat(cate_var_results_formatted_list).reset_index(drop=True)
#cate_results_formatted["Label"] = cate_results_formatted["CATE feature"] + " " + cate_results_formatted["Group"]

fp.forestplot(
    cate_results_formatted,  # the dataframe with resultcodes data
    estimate="CATE",  # col containing estimated effect size
    ll="lb", 
    hl="ub",  # columns containing conf. int. lower and higher limits
    varlabel="Group",  # column containing variable label
    ylabel="Confidence interval",  # y-label title
    xlabel=f"CATE on {OUTCOME2LABELS[outcome_name]}",  # x-label title
    # annote=["treatment_model", "event_aggregation"],  # columns to annotate
    groupvar="CATE_feature",  # group variable
    group_order=list(LABEL_MAPPING_HTE_BINARY_NAME.keys()),
    #figsize=(5, 12),
    color_alt_rows=True,
    sort=True,  # sort in ascending order (sorts within group if group is specified)               
)
plt.savefig(path2img / f"cate_forest_plot_{cohort_name}.pdf", bbox_inches="tight")
plt.savefig(path2img / f"cate_forest_plot_{cohort_name}.png", bbox_inches="tight")
