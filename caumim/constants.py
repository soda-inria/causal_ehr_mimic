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

# COlUMNS
# Colnames of the event table
COLNAME_CODE = "code"
COLNAME_LABEL = "name"
COLNAME_VALUE = "value"
COLNAME_CATEGORY = "category"
COLNAME_START = "starttime"
COLNAME_END = "endtime"
COLNAME_PATIENT_ID = "subject_id"
COLNAME_HADM_ID = "hadm_id"

# Other Colnames
COLNAME_INCLUSION_START = "inclusion_start"
COLNAME_FOLLOWUP_START = "followup_start"
COLNAME_TREATMENT_START = "treatment_start"
COLNAME_TREATMENT_STATUS = "treatment_status"


# legacy: keep?
# Expert features
STATIC_FEATURES_BASICS = ["age", "gender", "insurance"]
# Variables grouping derived from [Wang et al., 2020]() :
# We uncoment the "expert variables" :
EXPERT_VARIABLES = [
    # "Alanine aminotransferase",  # fonction rénale (grimpe), noraml de les faire une fois mais ensuite on arrête
    # "Albumin", # nutrition
    # "Alkaline phosphate",  # souffrance biliaire (tuyauterie), hépatique, très bon marqueur de trombose
    "Anion gap",  # equilibre acido-basique, marqueur pronostic, fonction rénale, (pas lié au scan, en fait si)
    # "Asparate aminotransferase",
    "Bicarbonate",  # pronostic, bilan anionique
    # "Bilirubin",  # bilan hépatique
    "Blood urea nitrogen",  # "fonctionnement rénale" : insuffisance rénale fonctionnelle ou organique selon son sens, mais pas hyper sensible -> internvetniton sur scan cereb
    # "CO2", # rarement fait
    # "CO2 (ETCO2, PCO2, etc.)", # marqueur prognostic
    "Calcium",  # fonctionnement renal + PH, pas parcequ'on a  ça qu'on va au scan
    "Calcium ionized",
    # "Central Venous Pressure",  # voie centrale : surveillance, pas discriminant pour le scanner
    "Chloride",  # anion gap
    # "Cholesterol",  # il monte pas de manière aigue, par contre on le fait plutôt en prév secondaire pr eviter la récidive
    "Creatinine",  # le rein filtre ou pas, Claire le garderait pas forcément
    # "Diastolic blood pressure", # Surveillance, n'influe pas sur le CTscan
    # "Fibrinogen",  # en Franc eon le fait bcp, marqueur prognostic de l'état de choc, molécule dégradée quand tu saignes -> indicateur d'hémorragie
    # "Fraction inspired oxygen", # mesure artérielle
    # "Fraction inspired oxygen Set", # mesure artérielle
    "Glascow coma scale total",  # Oui ! Concu pr le trauma cranien : énormement utilisé par le samu aux USA, en France fait tout le long de la chaîne
    # "Glucose", # Comme le cholésterol, facteur de risque donc on va vouloir traitier, mais quand on fait un AVC, elle est elevée de base donc il faut la respecter dans une certaine limite (eg. 2g), elle ne guide pas vers le scan
    # "Heart Rate", Ne guide pas
    # "Height",  # important pr l'IMC, facteur de risque, mais ne guide pas vers le scanner
    # "Hematocrit", # fraction de globule rouge dans un volume sanguin
    "Hemoglobin",  # charge en hemoglobin
    "Lactic acid",  # charge anionic pronostic
    # "Magnesium", # ne guide pas le ? fait en France
    # "Mean blood pressure", # surveillance, pourrait  ptetre
    "Oxygen saturation",  #
    # "Partial pressure of carbon dioxide", # caté veineux
    "Partial pressure of oxygen",  # jamais fait
    # "Partial thromboplastin time", en dehors de tte clinique, pas vraiment,
    # "Peak inspiratory pressure", # surveillance
    # "Phosphorous", # fonction rénale
    # "Plateau Pressure",
    "Platelets",  # plutot de la surveillance
    # "Positive end-expiratory pressure", # surveillance
    # "Positive end-expiratory pressure Set"
    "Potassium",  # bilan renal, hyper/hypocalémie tue,n ne guide pas le scan
    "Potassium serum",  # Quelle différence avec le Potassium ?: tube citraté, pr éviter le K de précipiter
    # "Prothrombin time INR", # bilan d'hemostase, c'est de l'éthiologie
    # "Prothrombin time PT" Deux façons différence de regarder le risque de saignement
    # "Pulmonary Artery Pressure systolic", # reflet de fonctionnement du coeur droit
    # "Red blood cell count", morpho des cellules rouges (nb: les femmes ont plus d'anémie, et des globules rouges plus petites)
    # "Respiratory rate", très monitoré pr surveillance
    # "Respiratory rate Set"
    # "Sodium", # fonction rénale, ions, contrôle du volume
    # "Systolic blood pressure", # surveillance
    "Temperature",  # surveillance, infection
    # "Tidal Volume Observed", # EFR, surveillance
    # "Tidal Volume Set"
    # "Tidal Volume Spontaneous"
    # "Total Protein", # bilan nutrition
    "Troponin-T",  # infarctus cardiaque, dégrade certaine protéine quand le coeur meurt
    # "Venous PvO2" # never used... no O2 in veines
    # "Weight",  # evolue bcp en réa
    "White blood cell count",  # marqueur d'infection si augmente, ou si diminue (chimio ou VIH)
    "pH",  # surveillance rea classique, anion gap
]

TOP_50_VARIABLES = [
    "Respiratory rate",
    "Heart Rate",
    "Oxygen saturation",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Mean blood pressure",
    "Temperature",
    "Central Venous Pressure",
    "Glucose",
    "Glascow coma scale total",
    "Positive end-expiratory pressure Set",
    "Tidal Volume Observed",
    "pH",
    "Peak inspiratory pressure",
    "Pulmonary Artery Pressure systolic",
    "Fraction inspired oxygen",
    "White blood cell count",
    "Potassium",
    "Fraction inspired oxygen Set",
    "Sodium",
    "Chloride",
    "Creatinine",
    "Blood urea nitrogen",
    "Magnesium",
    "Tidal Volume Spontaneous",
    "Respiratory rate Set",
    "Tidal Volume Set",
    "Hematocrit",
    "Phosphorous",
    "Calcium",
    "CO2 (ETCO2, PCO2, etc.)",
    "Partial pressure of carbon dioxide",
    "Partial thromboplastin time",
    "Prothrombin time INR",
    "Prothrombin time PT",
    "Platelets",
    "Hemoglobin",
    "Partial pressure of oxygen",
    "Pulmonary Artery Pressure mean",
    "Plateau Pressure",
    "Calcium ionized",
    "Weight",
    "Cardiac Index",
    "Lactic acid",
    "Systemic Vascular Resistance",
    "CO2",
    "Red blood cell count",
    "Cardiac Output Thermodilution",
    "Potassium serum",
    "Bicarbonate",
    "Height",
]

BASELINE_VARIABLES = [
    "Respiratory rate",
    "Oxygen saturation",
    "Heart Rate",
    "Mean blood pressure",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    # Glascow coma scale eye opening # not extracted in ehr cohort
    # Glascow coma scale verbal response
    # Glascow coma scale motor response
    "Glascow coma scale total",
    "Glucose",
    "Temperature",
    "Fraction inspired oxygen",
    "pH",
    "Weight",
    "Height"
    # capillary refill rate # no signal (only 1 in cohort and not extracted in ehr_cohort)
]
