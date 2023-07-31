
# Main study: Albumin for sepsis compared to crystalloids

The following notebooks implement our causal framework in a step-by-step fashion for didactic purpose for the targeted trial estimating the effect of *Albumin+crystalloids comapred to crystalloids only for sespis patients*:

- `_1_framing_albumin_for_sepsis.py`: Build the PICO(T) components for the targeted trial (Population, Intervention, Comparator, Outcome, Time).
- The function[`get_event_covariates_albumin_zhou`](https://github.com/soda-inria/causal_ehr_mimic/blob/06a65e94c221bd02d1613477937b281543b05577/caumim/variables/selection.py#L107) function in `caumim.variables.selection.py` implements in MIMIC-IV the selection of the confounders obtained during the identification step with the causal graph.

- `_3_estimation__albumin_for_sepsis.py`: Perform the estimation step with various causal and statistical models.

*NB:* The analyses of the paper were conducted with dedicated scripts present in `caumim.experiments`.

# Target trials proposal suitable to be replicated in MIMIC

The notebook prefixed with `apd__framing` contain the implementations of the target populations for possible emulated trials in MIMIC-IV. See *Appendix B - Estimation of Treatment effect with MIMIC data* in paper for details: 
 
 - Fludrocortisone combination for sepsis: Hydrocortisone vs both cortisones for sepsis patients

 - NBMA agent for ARDS: Neuromuscular blocking agent (NBMA) vs no NBMA for ARDS

 - Oxygenation for myocardial infarctus: Routine oxygen vs no oxygen for infarctus

 - Prone positioning for ARDS: Prone positioning vs supline positioning for ARDS


# Experimentations for data connections

- `syntaxes_for_data_connections.py` (LEGACY ONLY): Left for people wanting to experiment with other solutions: Explorations to handle the MIMIC data on a laptop. I tried different data processing packages and databases (duckdb, polars, postgresql) looking for rapidity and no out-of-memory errors. The final choice was: 
    - Built the data using postgresql;
    - Built the concepts from the postgresql concepts scripts;
    - Built and write to parquet the original database with duckdb in order to copy the large files to parquet without memory overflow;
    - Convert all derived tables from postgresql to parquet using polars
    - Do the analysis with polars.

# Technical details

## Plugin JupyTEXT

We use the [JupyTEXT](https://github.com/mwouts/jupytext) plugin to version notebooks.

Command to launch Jupyter correctly configured
```
make jupyter-notebook
```

Then, you can open either the paired files `.ipynb` or `.py`.
