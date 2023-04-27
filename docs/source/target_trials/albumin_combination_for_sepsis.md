# Albumin combination for sepsis 

# Notes

- Drugs are not well detected in the prescriptions tables: 

| intervention_status |     False |       True |
|:----------------|----------:|-----------:|
| Glycopeptide    | 0.209479  | 0.214654   |
| Beta-lactams    | 0.159558  | 0.120743   |
| Carbapenems     | 0.0192733 | 0.0154799  |
| Aminoglycosides | 0.007109  | 0.00515996 |

Compared to the paper from {cite:p}`zhou2021early` where target population prevalence are: 16.5% for Carbapenems, 87.3% for Glycopeptide, 28.3 for Beta-lactams, and 2.22 for Aminoglycosides. I got almost the same quantities when querrying the mimic_derived.antibiotic table directly which apply some filter on the routes.
They might have query in other tables such as emar or pharmacy ?


```{bibliography}
:filter: docname in docnames
```