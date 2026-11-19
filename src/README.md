# Package `src`

Ce répertoire contient le code source principal du projet (entraînement, évaluation/visualisation des runs et utilitaires).

## Organisation

```text
src/
├── __init__.py
├── results.py          # Génération de figures de métriques et résumé CSV depuis res/
├── umap_res.py         # Génération de figures UMAP par run (données + modèle)
├── z_res.py            # Histogrammes y_true / y_low / y_sup par run
├── training/           # Cœur entraînement: architectures, losses, métriques, pipeline, boucle
└── utils/              # Utilitaires partagés (config, plots, FITS)
```

## Modules racine

### `results.py`

Script de post-traitement des résultats d'entraînement.

- Entrée : fichiers `res/*.json`
- Sortie : `res_plots/<run>/...` et `res_plots/results_summary.csv`
- Fonctionnalités :
  - trace chaque métrique par dataset (avec séparation de la validation en figures dédiées quand pertinent)
  - génère une figure combinant toutes les métriques
  - agrège les dernières valeurs de métriques par run/dataset dans un CSV

Exécution :

```bash
python -m src.results
```

### `umap_res.py`

Script de visualisation de l'espace latent avec UMAP pour chaque run.

- Entrées :
  - paires `res/<run>.json` et `res/<run>.pth`
  - données latentes/cibles `data/epoch_40_train_results.npy` et `data/epoch_40_val_results.npy`
- Sortie : `umap_res/<run>/...`
- Fonctionnalités :
  - chargement et filtrage des splits train/validation via `EspaceLatent`
  - calcul d'un embedding UMAP sur le train, projection de la validation
  - prédiction des bornes d'intervalle via le modèle entraîné, avec la normalisation
    propre à chaque split (`EspaceLatent`, comme pendant l'évaluation en training)
  - figures de valeurs (`y_true`, `y_low`, `y_sup`) et figures d'erreur du centre

Exécution :

```bash
python -m src.umap_res
```

Options CLI (`--help`) : `--max-samples` (10000, 0 = tous), `--seed` (42),
`--n-neighbors` (15), `--min-dist` (0.1), `--batch-size` (512).

### `z_res.py`

Histogrammes superposés de `y_true`, `y_low` et `y_sup` pour le train et la validation.

- Réutilise `load_split`, `load_model` et `build_run_values` de `umap_res.py`
- Sortie : `z_res/<run>/<dataset>_histogram.png`
- Options CLI : `--max-samples` (10000, 0 = tous), `--seed` (42), `--batch-size` (512), `--bins` (60)

```bash
python -m src.z_res
```

## Sous-packages

### `training/`

Contient les composants nécessaires à l'entraînement :

- architectures MacSum et variantes
- fonctions de loss et métriques d'évaluation d'intervalles
- pipeline de chargement/préparation des données
- registres (modèles, optimizers, schedulers, losses, métriques) et script d'entraînement

Voir `src/training/README.md` pour le détail.

### `utils/`

Utilitaires transverses :

- `configs.py` : lecture de configuration TOML, récupération de device (`DEVICE`), chargement de modèle depuis `.pth`
- `plots.py` : helpers partagés par `src.results`, `src.umap_res` et `src.z_res` (chargement des runs/`EspaceLatent`, inférence `build_run_values`, sauvegarde de figures, tracés de métriques et histogrammes)
- `fits_management.py` : helpers pour extraire des données d'images depuis des fichiers FITS (`astropy`)

## Conventions générales

- Les chemins de données/résultats des scripts principaux sont définis par défaut relativement à la racine du projet (`data/`, `res/`, `res_plots/`, `umap_res/`).
- Les runs sont identifiés par un `run_name` commun aux fichiers `.json` (résultats/config) et `.pth` (poids).
- Les modèles d'intervalle produisent typiquement deux sorties (`y_low`, `y_sup`) consommées par les losses/métriques dédiées.
