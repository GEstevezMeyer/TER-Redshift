# Module `src.training`

Ce module regroupe les composants nécessaires pour entraîner et évaluer des modèles de prédiction d'intervalles (redshift) basés sur des architectures MacSum.

## Contenu

- `training.py` — point d'entrée CLI, registres, boucle d'entraînement/évaluation, sauvegarde des résultats
- `architecture.py` — briques et architectures (`NeuroneMacSum`, extensions, `MacSumDenseMinMax`, `ArchitectureMacSum`, `SimpleMacSum`)
- `loss.py` — fonctions de perte pour sorties ponctuelles ou par intervalle
- `metric.py` — métriques d'évaluation (Winkler, largeur, MAE/RMSE/IR/OOB, SigmaMAD, etc.)
- `pipeline.py` — chargement des données `.npy`, filtrage, normalisation, création des DataLoaders
- `point_estimator.py` — estimateur ponctuel simple à partir d'un intervalle
- `quantile.py` — baselines de régression quantile (historique, LightGBM, XGBoost) pour comparaison
- `__init__.py` — initialisation du module (détection CUDA/CPU et variable d'environnement `DEVICE`)

## Pipeline de données (`pipeline.py`)

Chemins par défaut :

- `data/epoch_40_train_results.npy`
- `data/epoch_40_val_results.npy`

Format attendu (objet dictionnaire dans le `.npy`) :

- `final_latent` : matrice/collection de features latentes
- `z_true` : cibles de redshift

Traitements appliqués :

- filtrage des cibles invalides (`z_true` fini et > 0)
- normalisation min-max des features et de la cible dans `[-1, 1]`
- conservation des statistiques de normalisation (`min`, `max`, `range`) sur le dataset

`main_pipeline(batch_size, test_size)` :

- charge train/validation
- découpe le dataset d'entraînement en sous-ensembles train/test via `random_split` (seed fixe)
- renvoie :
  - un `DataLoader` d'entraînement (shuffle)
  - un dictionnaire de `DataLoader` d'évaluation : `train`, `validation`, `test`

## Architectures (`architecture.py`)

### Briques principales

- `NeuroneMacSum` : neurone de base produisant une sortie d'intervalle `(y_low, y_sup)`
- `ConjunctiveExtension` / `DisjunctiveExtension` : extensions pour la propagation d'intervalles entre blocs
- `MacSumDenseMinMax` : bloc combinant MacSum, couches linéaires, activation, dropout et contrainte min/max
- `MinMax` : utilitaire garantissant l'ordre `(min, max)`

### Modèles enregistrables

- `ArchitectureMacSum` : architecture modulaire multi-blocs (paramétrée par listes d'hyperparamètres)
- `SimpleMacSum` : variante simple à un neurone MacSum (entrée -> intervalle)

Les activations disponibles sont définies dans `ACTIVATION_FUNCTION_REGISTERY` (`relu`, `leaky_relu`, `gelu`, `tanh`, etc.).

## Losses (`loss.py`)

Le module fournit plusieurs pertes, notamment :

- `RMSELoss` — RMSE classique
- `IntervalRMSELoss` — RMSE pour intervalles (`combined`, `separate`, `midpoint`)
- `IntervalQuantileLoss` — perte pinball (quantiles) + terme de centrage
- `WinklerLoss` — version différentiable du score de Winkler
- `SWRMSE_WS` — perte conditionnée à l'appartenance de la cible à l'intervalle
- `SWRMSE_RMSEb` — variante combinant RMSEb (inside) et RMSE centre (outside)
- `RMSEb` — erreur symétrique sur les deux bornes
- `WorstBound_RMSE` — pénalise la pire borne

## Métriques (`metric.py`)

Métriques/évaluateurs disponibles :

- `WinklerScore` — score de Winkler (alpha)
- `IntervalWidth` — largeur moyenne de l'intervalle
- `GeneralMetricsEvaluation` — retourne `(mae, rmse, oob_mae, oob_rmse, in_range_rate)`
- `SigmaMAD` — dispersion robuste des erreurs normalisées
- `Biais` — biais moyen des erreurs normalisées

## Registres et entraînement (`training.py`)

### Registres

- Modèles : `ArchitectureMacSum`, `SimpleMacSum`
- Optimiseurs : `sgd`, `adam`, `adamw`, `adamax`, `adagrad`, `adadelta`, `rmsprop`, `nadam`, `radam`, `lbfgs`
- Schedulers : `step` (custom), `steplr`, `multisteplr`, `exponentiallr`, `cosineannealinglr`
- Losses : ex. `mse`, `rmse`, `interval_rmse`, `intervalquantile`/`pinball`, `winkler_loss`, `SWRMSE_WS`, `SWRMSE_RMSEb`, `rmseb` (alias `rsmeb`), `WorstBound_RMSE`
- Métriques : `winkler`, `sigma_mad` (alias `sigmamad`)

Les fonctions `get_model`, `get_optimizer`, `get_scheduler`, `get_loss`, `get_metric` instancient les composants à partir des sections de configuration.

### Boucle d'entraînement

- seed fixe (`torch`/`numpy`)
- sélection du device via `utils.configs.return_device()` (fallback CPU)
- entraînement batch par batch avec gradient clipping (`max_norm=1.0`)
- évaluation périodique (`eval_freq`) sur `train`, `validation`, `test`
- métriques suivies par dataset : `loss`, `metrics`, `width`, `mae`, `rmse`, `ir`, `oob_mae`, `oob_rmse`

### Sauvegarde

`save_results(...)` écrit dans `res/` :

- `<run_name>.json` : historique (loss/lr/évaluations), tailles de datasets, config complète, paramètres du scheduler et normalisation
- `<run_name>.pth` : `state_dict` du modèle

### Lancement CLI

```bash
python -m src.training.training \
  --config configs/config_training_1.toml \
  --batch_size 32 \
  --epochs 10 \
  --eval_freq 1 \
  --test_size 0.2 \
  --run_name results
```

## Format de configuration (TOML)

### Exemple `ArchitectureMacSum`

```toml
[model]
model_name = "ArchitectureMacSum"
input_shape = 1024
lis_MacSumOut = [128, 64, 32]
lis_output_size = [32, 16, 1]
lis_activation = ["leaky_relu", "leaky_relu", "leaky_relu"]
lis_increments = [2, 1, 1]
lis_linears_start = [128, 32, 16]
lis_LinearN = [3, 2, 1]
p_dropout = 0.0

[training.optimizer]
name = "adamw"
lr = 0.001
weight_decay = 0.01

[training.scheduler]
name = "step"
gamma = 0.1
min_lr = 0.00001

[training.loss]
name = "SWRMSE_RMSEb"
eps = 0.00001

[training.metric]
name = "winkler"
alpha = 0.1
```

### Exemple `SimpleMacSum`

```toml
[model]
model_name = "SimpleMacSum"
input_size = 1024
```

Les clés supplémentaires des sections `loss`, `metric`, `optimizer` et `scheduler` sont transmises aux constructeurs/classes correspondants (selon le registre).

## Baselines quantiles (`quantile.py`)

Le fichier `quantile.py` fournit des baselines hors pipeline principal :

- `BaselineHistorique` : intervalle fixe basé sur les quantiles empiriques de `y_train`
- `BaselineLightGBM` : deux modèles quantiles (borne basse/haute)
- `BaselineXGBoost` : régression quantile multi-sortie

Utile pour comparaison expérimentale, mais non intégré au CLI `src.training.training`.
