# %%
import pandas as pd
from caumim.constants import *

# %%
# conversion to parquet of [the ndc_map](https://github.com/fabkury/ndc_map/blob/master/FDA%20NDC%20Directory%20with%20drug%20classes/FDA%20NDC%20directory%20with%20atc5%20atc4%20ingredients%20(2020_06_17).zip)
ndc_map = pd.read_csv(
    DIR2RESOURCES
    / "ontology"
    / "ndc_map 2020_06_17 (atc5 atc4 ingredients).csv"
)
ndc_map.to_parquet(DIR2RESOURCES / "ontology" / "ndc_map")
