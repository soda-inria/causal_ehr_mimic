# Target trials proposal suitable to be replicated in MIMIC

This folder contains the implementation of target population for possible emulated trials in MIMIC-IV. See dedicated Appendix in paper for details. 

1 - Albumin for sepsis: Crystalloids vs Crystalloids + Albumin for sepsis patients

2 - Fludrocortisone combination for sepsis: Hydrocortisone vs both cortisones for sepsis patients

3 - NBMA agent for ARDS: Neuromuscular blocking agent (NBMA) vs no NBMA for ARDS

4 - Oxygenation for myocardial infarctus: Routine oxygen vs no oxygen for infarctus

5 - Prone positioning for ARDS: Prone positioning vs supline positioning for ARDS


# Technical details

## Plugin JupyTEXT

We use the [JupyTEXT](https://github.com/mwouts/jupytext) plugin to version notebooks.

Command to launch Jupyter correctly configured
```
make jupyter-notebook
```

Then, you can open either the paired files `.ipynb` or `.py`.

