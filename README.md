# TER-Redshift

Projet de recherche (TER) sur l'estimation de redshift à partir de représentations latentes, avec des architectures de type **MacSum** pour la prédiction d'intervalles.

## Objectif

L'objectif est d'entraîner et d'évaluer des modèles capables de prédire un intervalle `[y_low, y_sup]` autour du redshift cible (`z_true`) à partir de vecteurs latents, puis d'analyser les performances (métriques d'intervalle, couverture, erreurs) et la structure de l'espace latent (UMAP).

## Structure du dépôt

- `src/` — code source du projet
  - `src/training/` — modèles, losses, métriques, pipeline de données et boucle d'entraînement
  - `src/utils/` — utilitaires (configuration, tracés de résultats, gestion FITS)
  - `src/results.py` — génération de figures et d'un résumé CSV à partir des résultats sauvegardés
  - `src/umap_res.py` — génération de visualisations UMAP pour chaque run sauvegardé
- `configs/` — configurations d'entraînement au format TOML
- `data/` — jeux de données (`.npy`) utilisés par le pipeline
- `res/` — sorties d'entraînement (`<run>.json`, `<run>.pth`)
- `res_plots/` — figures générées par `src.results`
- `umap_res/` — figures UMAP générées par `src.umap_res`
- `notebooks/` — notebooks d'exploration/visualisation
- `test/` — tests
- `requirements.txt` — dépendances Python

## Prérequis

- Python (environnement avec les dépendances installées)
- Dépendances listées dans `requirements.txt` (PyTorch, NumPy, Matplotlib, UMAP, etc.)

Exemple d'installation :

```bash
pip install -r requirements.txt
```

## Données attendues

Le pipeline de données (`src/training/pipeline.py`) lit par défaut :

- `data/epoch_40_train_results.npy` (jeu d'entraînement)
- `data/epoch_40_val_results.npy` (jeu de validation)

Chaque fichier `.npy` doit contenir un dictionnaire sérialisé avec au moins :

- `final_latent` : features latentes (vecteurs)
- `z_true` : redshift cible

Les échantillons invalides (`z_true` non fini ou <= 0) sont filtrés. Les features et la cible sont normalisées en min-max dans `[-1, 1]`.

## Entraînement

Lancer un entraînement depuis la racine du projet :

```bash
python -m src.training.training --config configs/config_training_1.toml --epochs 10 --eval_freq 1 --batch_size 32 --test_size 0.2 --run_name results
```

Arguments principaux :

- `--config` : chemin du fichier TOML de configuration (obligatoire)
- `--epochs` : nombre d'époques
- `--eval_freq` : fréquence d'évaluation (en époques)
- `--batch_size` : taille de batch
- `--test_size` : proportion du jeu d'entraînement réservée au split test (défaut : 0.2)
- `--run_name` : nom des fichiers de sortie dans `res/`

Sorties produites :

- `res/<run_name>.json` : historique d'entraînement/évaluation, configuration, tailles de splits, métadonnées de scheduler et normalisation (`x_min/x_max/y_min/y_max`)
- `res/<run_name>.pth` : poids du modèle (`state_dict`)

Les jeux évalués pendant l'entraînement sont `train`, `validation` et `test` (le split test est extrait du jeu d'entraînement).

## Visualisation des résultats

Générer les courbes de métriques et le résumé CSV pour tous les runs présents dans `res/` :

```bash
python -m src.results
```

Sorties dans `res_plots/` :

- figures par métrique et figures combinées (par run)
- séparation de la validation en figures dédiées lorsque plusieurs datasets sont présents
- `res_plots/results_summary.csv` (valeurs finales par métrique/dataset et noms de loss/métrique)

## Visualisation UMAP

Générer les projections UMAP des espaces latents et les prédictions d'intervalle pour chaque paire `res/<run>.json` + `res/<run>.pth` :

```bash
python -m src.umap_res
```

Options (`python -m src.umap_res --help`), avec les valeurs par défaut entre parenthèses :
`--max-samples` (10000, 0 = tous les échantillons), `--seed` (42),
`--n-neighbors` (15), `--min-dist` (0.1), `--batch-size` (512).

Sorties dans `umap_res/<run>/` :

- `<dataset>_umap.png` : panneaux `y_true`, `y_low`, `y_sup`
- `<dataset>_umap_error.png` : erreur du centre prédit `y_true - (y_low + y_sup) / 2`

Les prédictions passent par le `Dataset` `EspaceLatent` de `training.pipeline` : chaque split est normalisé avec ses propres statistiques, comme pendant l'évaluation en training (y compris pour la validation).

## Histogrammes des prédictions

Générer les distributions de `y_true`, `y_low` et `y_sup` pour chaque run avec le même pipeline d'inférence que `umap_res` :

```bash
python -m src.z_res
```

Options (`python -m src.z_res --help`) : `--max-samples` (10000, 0 = tous),
`--seed` (42), `--batch-size` (512), `--bins` (60).

Sorties dans `z_res/<run>/` : `<dataset>_histogram.png` (trois courbes superposées sur des bins communs).

## Configuration

Les fichiers TOML définissent notamment :

- `[model]` : architecture (`model_name`) et ses hyperparamètres
- `[training.optimizer]` : optimiseur et paramètres associés
- `[training.scheduler]` : scheduler (optionnel selon config)
- `[training.loss]` : fonction de loss
- `[training.metric]` : métrique principale suivie pendant l'évaluation

Exemples disponibles dans `configs/` (`config_training_1.toml`, `config_training_simple.toml`, `config_training_pinball.toml`, etc.).

## Tests

```bash
pytest
```

## Documentation complémentaire

- `src/README.md` — vue d'ensemble des modules du package `src`
- `src/training/README.md` — détails du module d'entraînement (architectures, registres, pipeline, configuration)
