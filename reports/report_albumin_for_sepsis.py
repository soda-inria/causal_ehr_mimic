import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from caumim.constants import *
from caumim.framing.albumin_for_sepsis import COHORT_CONFIG_ALBUMIN_FOR_SEPSIS
from caumim.framing.utils import create_cohort_folder

# %%
cohort_dir = create_cohort_folder(COHORT_CONFIG_ALBUMIN_FOR_SEPSIS)
# %%
results = pd.read_parquet(cohort_dir / "estimates")
results["label"] = (
    results["event_aggregations"].map(
        lambda x: x.split(".")[-1].replace("()", "")
    )
    + ""
    + results["estimation_methods"]
    + "\n"
    + results["treatment_model"]
)
# %%
# zepid: bof bof...
from zepid.graphics.graphics import EffectMeasurePlot, zipper_plot

f_plot = EffectMeasurePlot(
    label=results["label"],
    effect_measure=results[RESULT_ATE],
    lcl=results[RESULT_ATE_LB],
    ucl=results[RESULT_ATE_UB],
)
f_plot.plot(figsize=(6.5, 3), t_adjuster=0.1, min_value=-0.15, max_value=0.15)
# plt.tight_layout()
plt.show()
# plt.savefig(DIR2DOCS_IMG / "albumin_for_sepsis_forest_plot.pdf")
# %%
import forestplot as fp

fp.forestplot(
    results,  # the dataframe with results data
    estimate=RESULT_ATE,  # col containing estimated effect size
    ll=RESULT_ATE_LB,
    hl=RESULT_ATE_UB,  # columns containing conf. int. lower and higher limits
    varlabel="label",  # column containing variable label
    ylabel="Confidence interval",  # y-label title
    xlabel="ATE",  # x-label title
)
