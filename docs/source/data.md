# Data documentation
Use this to document the data used in the project.

We recommend using
[panderas](https://pandera.readthedocs.io/en/stable/) to document
data structures where possible.

As a minimal documentation process, we suggest:
- Use the
[infer_schema](https://pandera.readthedocs.io/en/stable/schema_inference.html)
method to create the simpliest schema for your tables, containing inferred types
and column names.
- Store the schemas of each table as dedicated `.json` files in a `schemas` folder at
  the root of the package.
- Enrich the schemas by hand all along the project.

Please also document:
- the data provider:
  - where did you get this data ?
  - who is responsible for providing it ?
  - When did you get it ?
- the data life-cycle:
  - refresh frequency
  - Storage location
- Data outputs
