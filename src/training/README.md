# Training Module

## Files
- `architecture.py` - MacSum neural network architectures
- `training.py` - Training loop
- `loss.py` - Loss functions (RMSE, IntervalRMSE, IntervalQuantile, Winkler, SWRMSE_WS, SWRMSE_RMSEb, RMSEb, WorstBound_RMSE)
- `metric.py` - Evaluation metrics (WinklerScore, GeneralMetricsEvaluation, IntervalWidth, SigmaMAD)
- `pipeline.py` - Data loading and preprocessing
- `point_estimator.py` - Point estimation layers



## Usage
Use configs in `configs/` to train models. The `[model]` section must start with
`model_name`, which selects the class from `MODEL_REGISTRY` in
`src/training/training.py`; the remaining keys are passed to its constructor:

```toml
[model]
model_name = "ArchitectureMacSum"
input_shape = 1024
lis_MacSumOut = [128, 64, 32]
```

To use another architecture, define it in `architecture.py`, add one entry to
`MODEL_REGISTRY`, and set `model_name` to that key in the config. The same
`model_name` is stored in `res/*.json`, so saved models can be reloaded with
`get_model` (used by `training.py`, `umap_res.py` and `utils/configs.py`).
