# %%
from matplotlib import pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from caumim.constants import *
from caumim.experiments.utils import make_column_transformer
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder
from caumim.inference.utils import make_random_search_pipeline

from caumim.variables.selection import get_event_covariates_albumin_zhou
from caumim.variables.utils import (
    feature_emergency_at_admission,
    feature_insurance_medicare,
    get_measurement_from_mimic_concept_tables,
)

from sklearn import clone
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV

from dowhy import CausalModel
from caumim.inference.estimation import AteEstimator, CateEstimator

# %%
# 1 - Framing
cohort_folder = create_cohort_folder(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
target_trial_population = pl.read_parquet(
    cohort_folder / FILENAME_TARGET_POPULATION
)
target_trial_population.head()
# %%
# 2 - Variable selection

# Static features
# demographics
target_trial_population = feature_emergency_at_admission(
    target_trial_population
)
target_trial_population = feature_insurance_medicare(
    target_trial_population
).with_columns(
    [
        pl.when(pl.col("gender") == "F")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("Female"),
        pl.when(pl.col("race").str.to_lowercase().str.contains("white"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("White"),
    ]
)
static_features = [
    "admission_age",
    "Female",
    COLNAME_EMERGENCY_ADMISSION,
    COLNAME_INSURANCE_MEDICARE,
]
outcome_name = COLNAME_MORTALITY_28D
# %%
# event features
event_features, feature_types = get_event_covariates_albumin_zhou(
    target_trial_population
)


# %%
aggregate_functions = {
    "first": pl.col(COLNAME_VALUE).first(),
    "last": pl.col(COLNAME_VALUE).last(),
}
aggregate_functions = {k: v.alias(k) for k, v in aggregate_functions.items()}
aggregation_names = list(aggregate_functions.keys())
patient_features_aggregated = (
    event_features.sort([COLNAME_PATIENT_ID, COLNAME_START])
    .groupby(STAY_KEYS + [COLNAME_CODE])
    .agg(list(aggregate_functions.values()))
).pivot(
    index=STAY_KEYS,
    columns=COLNAME_CODE,
    values=list(aggregation_names),
)
event_features_names = list(
    set(patient_features_aggregated.columns).difference(set(STAY_KEYS))
)
X_list = patient_features_aggregated.join(
    target_trial_population,
    on=STAY_KEYS,
    how="inner",
)[
    [
        *event_features_names,
        *static_features,
        COLNAME_INTERVENTION_STATUS,
        outcome_name,
    ]
].to_pandas()
colnames_binary_features = [
    f"{agg_name_}_code_{col}"
    for agg_name_ in aggregation_names
    for col in feature_types.binary_features
]
colnames_numerical_features = [
    f"{agg_name_}_code_{col}"
    for agg_name_ in aggregation_names
    for col in feature_types.numerical_features
]
colnames_categorical_features = [
    f"{agg_name_}_code_{col}"
    for agg_name_ in aggregation_names
    for col in feature_types.categorical_features
]
X_list[colnames_binary_features] = X_list[colnames_binary_features].fillna(
    value=0
)
column_transformer = make_column_transformer(
    numerical_features=colnames_numerical_features,
    categorical_features=colnames_categorical_features,
)
# preview the feature preprocessing (imputation, categories handling and standardization)
column_transformer_preview = clone(column_transformer)
transformed_features_preview = column_transformer_preview.fit_transform(X_list)
transformed_features_preview = pd.DataFrame(
    transformed_features_preview,
    columns=column_transformer_preview.get_feature_names_out(),
)
transformed_features_preview.head()
# %% [markdown] Are we happy with the features preprocessing ? Note that the
# column_transformer should be apply in a sklearn.pipeline for the treatment or
# the outcome models in order to avoid information leakage.

# 3 - Identification and Estimation

## Let's take an identification and an estimation method
# To keep it simple, we will use a regularized logistic/linear regressions for both
# the treatment and outcome models, and will vary the identification methods.
# %%
from caumim.experiments.configurations import ESTIMATOR_RIDGE, ESTIMATOR_RF

estimator = ESTIMATOR_RIDGE
# %% [markdown] Because, there is some hyper-parameters to choose, we will use a
# random search to find the best hyper-parameters for our dataset, as
# recommended by [Bouthillier et al., 2021](https://arxiv.org/pdf/2103.03098.pdf). Then, for the
# different identification methods, we will reuse these hyper-parameters to fit
# the nuisance models of the outcome and the treatment.

treatment_pipeline = make_random_search_pipeline(
    estimator=estimator["treatment_estimator"],
    column_transformer=column_transformer,
    param_distributions=estimator["treatment_param_distributions"],
)
treatment_pipeline
# %%
outcome_pipeline = make_random_search_pipeline(
    estimator=estimator["outcome_estimator"],
    column_transformer=column_transformer,
    param_distributions=estimator["outcome_param_distributions"],
)
outcome_pipeline
# %%
a = X_list[COLNAME_INTERVENTION_STATUS]
y = X_list[outcome_name]
treatment_pipeline.fit(X_list, a)
treatment_estimator_w_best_HP = treatment_pipeline.best_estimator_
outcome_pipeline.fit(X_list, y)
outcome_estimator_w_best_HP = outcome_pipeline.best_estimator_
# %%
# ### G-computation with T-learner
from econml.metalearners import TLearner
from econml.inference import BootstrapInference

# NB: econml methods does not support missing data in the inputs, so we'll
# have to preprocess the data before fitting to the estimator. A practice to be
# avoided usually, because it can lead to [information leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)).
X_transformed = column_transformer.fit_transform(
    X_list.drop([COLNAME_INTERVENTION_STATUS, outcome_name], axis=1)
)
X_transformed = pd.DataFrame(
    X_transformed, columns=column_transformer.get_feature_names_out()
)
t_learner = TLearner(
    models=[
        outcome_estimator_w_best_HP.named_steps["estimator"],
        outcome_estimator_w_best_HP.named_steps["estimator"],
    ]
)
# The bootstrap inference method allows to compute CI for non-parametric estimators.
# However, it is computationally expensive and some doubts have been casted on the .
t_learner.fit(
    y, a, X=X_transformed, inference=BootstrapInference(n_bootstrap_samples=10)
)
results = {}
ate_inference = t_learner.ate_inference(X=X_transformed)
results[RESULT_ATE] = ate_inference.mean_point
results[RESULT_ATE_LB], results[RESULT_ATE_UB] = ate_inference.conf_int_mean()
results
# These are big error bounds: a hint indicating that we should not trust this T-learner.
# %% [markdown]
# ### AIPW estimator with doubly-robust inference
# %%
from econml.dr import LinearDRLearner

dr_learner = LinearDRLearner(
    model_propensity=treatment_estimator_w_best_HP.named_steps["estimator"],
    model_regression=outcome_estimator_w_best_HP.named_steps["estimator"],
    min_propensity=0.001,
    cv=5,
)
dr_learner.fit(
    y,
    a,
    X=None,
    W=X_transformed,
    inference=BootstrapInference(n_bootstrap_samples=10),
)
results = {}
ate_inference = dr_learner.ate_inference(X=None)
results[RESULT_ATE] = ate_inference.mean_point
results[RESULT_ATE_LB], results[RESULT_ATE_UB] = ate_inference.conf_int_mean()
results

# %% [markdown]
## With dowhy

# I find the package a bit too complicated if we are solely interested in ate
# estimation, but it implements a IPW out of box (which could overfit though because not fitted by crossvaldaito).
#  %%
dowhy_X = pd.concat([X_transformed, a, y], axis=1)
common_causes = list(
    dowhy_X.columns.drop([outcome_name, COLNAME_INTERVENTION_STATUS])
)
model = CausalModel(
    data=dowhy_X,
    treatment=COLNAME_INTERVENTION_STATUS,
    outcome=outcome_name,
    common_causes=common_causes,
)
identified_estimand = model.identify_effect(
    optimize_backdoor=True, proceed_when_unidentifiable=True
)

# treatment_pipeline = make_pipeline(*[column_transformer, treatment_estimator])
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",
    method_params={
        "propensity_score_model": treatment_estimator_w_best_HP["estimator"],
        "min_ps_score": 0.001,
        "max_ps_score": 0.999,
    },
    confidence_intervals=False,
)
lower_bound, upper_bound = estimate.get_confidence_intervals()
results = {}
results[RESULT_ATE] = estimate.value
results[RESULT_ATE_LB] = lower_bound
results[RESULT_ATE_UB] = upper_bound
results
# %%
# Use a Causal Forest
from econml.grf import CausalForest

transformed_data = column_transformer.fit_transform(
    X_list.drop([COLNAME_INTERVENTION_STATUS, outcome_name], axis=1),
)
transformed_data = pd.DataFrame(
    transformed_data, columns=column_transformer.get_feature_names_out()
)
forest_learner = CausalForest()
forest_learner.fit(
    X=transformed_data,
    y=X_list[outcome_name],
    T=X_list[COLNAME_INTERVENTION_STATUS],
)
(
    ate_point_estimates,
    lb_point_estimates,
    ub_point_estimates,
) = forest_learner.predict(X=transformed_data, interval=True)
results = {}
results[RESULT_ATE] = ate_point_estimates.mean()
results[RESULT_ATE_LB] = lb_point_estimates.mean()
results[RESULT_ATE_UB] = ub_point_estimates.mean()
results
# %%
# Ortho-learner
from econml.dml import LinearDML

dml_learner = LinearDML(
    model_t=treatment_estimator_w_best_HP["estimator"],
    model_y=outcome_estimator_w_best_HP["estimator"],
    # min_propensity=0.001,
    discrete_treatment=True,
    cv=5,
)
dml_learner.fit(
    y,
    a,
    X=None,
    W=X_transformed,
    inference=BootstrapInference(n_bootstrap_samples=10),
)
results = {}
ate_inference = dr_learner.ate_inference(X=None)
results[RESULT_ATE] = ate_inference.mean_point
results[RESULT_ATE_LB], results[RESULT_ATE_UB] = ate_inference.conf_int_mean()
results

# %%
# Naive DM estimate:
from zepid import RiskDifference

# The bound returned are the worst cases scenario from the [Fréchet-Boole
# inequalities](http://causality.cs.ucla.edu/blog/index.php/2019/11/05/frechet-inequalities/).
# Any estimator having larger bound than the naive DM estimator should not be
# reliable.
dm = RiskDifference()
dm.fit(X_list, COLNAME_INTERVENTION_STATUS, outcome_name)
dm_results = {
    RESULT_ATE: dm.results.RiskDifference[1],
    RESULT_ATE_LB: dm.results.LowerBound[1],
    RESULT_ATE_UB: dm.results.UpperBound[1],
}
dm_results

# %% [markdown]
# Check assumptions

## Graphical assessment
# %%
hat_e = cross_val_predict(
    estimator=treatment_estimator_w_best_HP,
    X=X_list,
    y=a,
    n_jobs=-1,
    method="predict_proba",
)[:, 1]
hat_analysis_df = pd.DataFrame(
    np.vstack([hat_e, a.values]).T,
    columns=[LABEL_PS, LABEL_TREATMENT],
)
hat_analysis_df[LABEL_TREATMENT] = hat_analysis_df.apply(
    lambda x: TREATMENT_LABELS[int(x[LABEL_TREATMENT])], axis=1
)

# %%
fix, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
    ax=ax,
    data=hat_analysis_df,
    x=LABEL_PS,
    hue=LABEL_TREATMENT,
    stat="probability",
    common_norm=False,
    palette=TREATMENT_PALETTE,
)

# %%
