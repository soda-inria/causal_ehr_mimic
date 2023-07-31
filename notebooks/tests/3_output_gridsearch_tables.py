# %%
from caumim.experiments.configurations import *
from caumim.constants import *
import pandas as pd

pd.options.display.max_colwidth = 100


# %%
def clean_grid(grid, scientific=True):
    grid_ = {}
    for k, v in grid.items():
        if scientific:
            v_scientific = [f"{x:.2E}" for x in v]
        else:
            v_scientific = [f"{x}" for x in v]
        grid_[k.replace("estimator__", "")] = v_scientific
    return grid_


ridge_gridsearch = ESTIMATOR_RIDGE
ridge_gridsearch_df = pd.DataFrame(
    {
        "Estimator type": ["Linear"] * 2,
        "estimator": [
            str(ridge_gridsearch["treatment_estimator"]).replace("()", ""),
            str(ridge_gridsearch["outcome_estimator"]).replace("()", ""),
        ],
        "nuisance": ["treatment", "outcome"],
        "Grid": [
            ridge_gridsearch["treatment_param_distributions"],
            ridge_gridsearch["outcome_param_distributions"],
        ],
    }
)
ridge_gridsearch_df["Grid"] = ridge_gridsearch_df["Grid"].apply(
    lambda x: clean_grid(x)
)
rf_gridsearch_df = pd.DataFrame(
    {
        "Estimator type": ["Forest"] * 2,
        "estimator": [
            str(ESTIMATOR_RF["treatment_estimator"]).replace("()", ""),
            str(ESTIMATOR_RF["outcome_estimator"]).replace("()", ""),
        ],
        "nuisance": ["treatment", "outcome"],
        "Grid": [
            ESTIMATOR_RF["treatment_param_distributions"],
            ESTIMATOR_RF["outcome_param_distributions"],
        ],
    }
)
rf_gridsearch_df["Grid"] = rf_gridsearch_df["Grid"].apply(
    lambda x: clean_grid(x, scientific=False)
)

gridsearch_df = pd.concat([ridge_gridsearch_df, rf_gridsearch_df])
gridsearch_df = gridsearch_df.set_index("Estimator type")
gridsearch_df.to_latex(DIR2DOCS_IMG / "gridsearch.tex")
