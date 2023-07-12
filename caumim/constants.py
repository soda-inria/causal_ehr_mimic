import os
from pathlib import Path

from dotenv import load_dotenv

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

load_dotenv()

ROOT_DIR = Path(
    os.getenv(
        "ROOT_DIR", Path(os.path.dirname(os.path.abspath(__file__))) / ".."
    )
)

# Default paths
# Data
DIR2DATA = ROOT_DIR / "data"
DIR2RESOURCES = DIR2DATA / "resources"
DIR2META_CONCEPTS = DIR2RESOURCES / "meta_concepts"
DIR2COHORT = DIR2DATA / "cohort"
DIR2EXPERIENCES = DIR2DATA / "experiences"
DIR2RESULTS = DIR2DATA / "results"
DIR2MIMIC = DIR2DATA / "mimiciv_as_parquet"

# Docs
DIR2DOCS = ROOT_DIR / "docs/source"
DIR2DOCS_STATIC = DIR2DOCS / "_static"
DIR2DOCS_IMG = DIR2DOCS_STATIC / "img"
DIR2DOCS_COHORT = DIR2DOCS_IMG / "cohort"
DIR2DOCS_EXPERIENCES = DIR2DOCS_IMG / "experiences"
DIR2DOCS_COHORT.mkdir(parents=True, exist_ok=True)
DIR2DOCS_EXPERIENCES.mkdir(parents=True, exist_ok=True)

# Default file names
FILENAME_TARGET_POPULATION = "target_population"
FILENAME_INCLUSION_CRITERIA = "inclusion_criteria.pkl"

# COlUMNS
# Colnames of the event table
COLNAME_PATIENT_ID = "subject_id"
COLNAME_HADM_ID = "hadm_id"
COLNAME_ICUSTAY_ID = "stay_id"
COLNAME_CODE = "code"
COLNAME_LABEL = "label"
COLNAME_VALUE = "value"
COLNAME_DOMAIN = "domain"
COLNAME_START = "starttime"
COLNAME_END = "endtime"
COLNAMES_EVENTS = [
    COLNAME_PATIENT_ID,
    COLNAME_HADM_ID,
    COLNAME_ICUSTAY_ID,
    COLNAME_START,
    COLNAME_END,
    COLNAME_DOMAIN,
    COLNAME_CODE,
    COLNAME_LABEL,
    COLNAME_VALUE,
]
STAY_KEYS = [COLNAME_PATIENT_ID, COLNAME_ICUSTAY_ID, COLNAME_HADM_ID]


# Other Colnames
COLNAME_INCLUSION_START = "inclusion_start"
# COLNAME_FOLLOWUP_START = "followup_start"
# By design I am forcing the FOLLOWUP_START to be the inclusion start. This should
# avoid making time-zero bias errors but might not be super practical.
COLNAME_INTERVENTION_START = "intervention_start"
COLNAME_INTERVENTION_STATUS = "intervention_status"
COLNAME_MORTALITY_28D = "mortality_28days"
COLNAME_MORTALITY_90D = "mortality_90days"
# delta
COLNAME_DELTA_MORTALITY = "delta mortality to inclusion"
COLNAME_DELTA_INTERVENTION_INCLUSION = "delta intervention to inclusion"
COLNAME_DELTA_INCLUSION_INTIME = "delta inclusion to intime"
COLNAME_DELTA_INTIME_ADMISSION = "delta ICU intime to hospital admission"

# Features
COLNAME_EMERGENCY_ADMISSION = "Emergency admission"
COLNAME_INSURANCE_MEDICARE = "Insurance, Medicare"

# Results columns
RESULT_ATE = "ATE"
RESULT_ATE_LB = "ATE lower bound"
RESULT_ATE_UB = "ATE upper bound"

# experiment constants
RANDOM_STATE = 42
MIN_PS_SCORE = 0.001

# report constants

OUTCOME2LABELS = {
    COLNAME_MORTALITY_28D: "28-day mortality",
    COLNAME_MORTALITY_90D: "90-day mortality",
}
LABEL_PS = "Estimated propensity score"
LABEL_TREATMENT = "Treatment status"
TREATMENT_LABELS = {0: "Control", 1: "Intervention"}
TREATMENT_PALETTE = {
    "Control": "#ff7f0e",
    "Intervention": "#1f77b4",
}

# Caironi et al. 2014, effect size (ATE)

LABEL_RCT_GOLD_STANDARD_ATE = "RCT Gold Standard (Caironi et al. 2014)"
# Computation of the confidence interval for risk difference is done with pooled variance

VALUE_RCT_GOLD_STANDARD_ATE = 0
VALUE_RCT_GOLD_STANDARD_ATE_LB = VALUE_RCT_GOLD_STANDARD_ATE
VALUE_RCT_GOLD_STANDARD_ATE_UB = VALUE_RCT_GOLD_STANDARD_ATE


IDENTIFICATION2LABELS = {
    "Difference in mean": "Unajusted risk difference",
    LABEL_RCT_GOLD_STANDARD_ATE: LABEL_RCT_GOLD_STANDARD_ATE,
    "backdoor.propensity_score_matching": "Propensity Score Matching",
    "backdoor.propensity_score_weighting": "Inverse Propensity Weighting",
    "TLearner": "Outcome model (TLearner)",
    "CausalForest": "Causal Forest",
    "LinearDML": "Double Machine Learning",
    "LinearDRLearner": "Doubly Robust (AIPW)",
}
