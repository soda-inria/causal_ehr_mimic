# Notebooks Jupyter

Les notebooks sont contenus dans le dossier `caumim/notebooks`.

Ils peuvent correspondre à des travaux exploratoires, d'analyse interactive ou de restitution.

La convention de nommage est un nombre (pour l'ordre), les initiales du créateur,
et une description délimitée par des tirets, e.g. `001-paj-exploration-données`.

## Plugin JupyTEXT

Nous utilisons le plugin [JupyTEXT](https://github.com/mwouts/jupytext) pour versionner les notebooks.

Commande pour lancer Jupyter correctement configuré

```
make jupyter-notebook
```

Vous pouvez alors ouvrir indifférement les fichiers `.ipynb` ou `.py` appairés.

### Pourquoi ce plugin ?

Les fichiers bruts des notebooks Jupyter (extension `.ipynb`) contiennent les résultats des calculs.
Ces résultats sont potentiellement sensibles, il faut donc éviter de les versionner.

JupyTEXT permet de sauvegarder le contenu des notebooks, sans ces sorties, dans un fichier python (extension `.py`).
Ces fichiers peuvent être versionnés sans risque.

De plus la différence entre 2 versions successives (`git diff`) est exploitable pour suivre l'évolution du code,
ce qui n'est pas le cas pour le format d'origine `.ipynb`.

Enfin, ces fichiers peuvent être rééxecutés facilement s'ils produisent des résultats,
par exemple `python 001-decouverte_données.py`.

Pour versionner un notebook *avec* ses sorties, par exemple s'il sert de rapport, préférer un export `pdf` ou `html`.

### Tracking dans git
Le dossier notebook est configuré pour ignorer par défaut tous les fichiers sauf les `.py` générés par jupytext.

### Dépannage

Parfois, l'un des 2 fichiers `.ipynb` et `.py` est modifié en dehors de Jupyter + JupyTEXT, via git ou une édition directe.

*Problème* : Les fichiers ne sont alors plus compatibles, ce qui créé une erreur à l'ouverture.

*Solution* : Supprimer (ou déplacer ailleurs par sécurité) l'un des deux fichiers.

## Gestion des data
Pour gérér l'accès aux données, une bonne pratique est d'utiliser une variable d'env qui pointe vers le dossier qui contient toute la donnée. De cette façon, on peut configurer ce chemin sans toucher au notebook.

```python
import os
import pathlib

DATA_PATH = pathlib.Path(os.getenv('DATA_PATH', '../data'))
```
