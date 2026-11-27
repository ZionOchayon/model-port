# Example: Covertype classifier

An XGBoost classifier for the forest cover type of a 30 m cell, from cartographic variables (UCI Covertype), trained with Ray Train's `XGBoostTrainer` on 2 workers.

```bash
modelport validate examples/models/covertype-classifier
modelport submit examples/models/covertype-classifier --wait
```

## How it meets the contract

- The signature has the 10 numeric variables plus `wilderness_area` (0-3) and `soil_type` (0-39). `examples/datasets/prepare.py` turns the dataset's 44 one-hot columns into these two indices.
- `preprocess` builds an `xgboost.DMatrix`. `predict` returns the most likely of the seven cover types with its probability (`multi:softprob`).
- `train_loop` maps the label names to 0-6, trains with `tree_method: hist`, and checkpoints every boosting round with `modelport.training.xgboost_callback()`. For XGBoost, one boosting round counts as one epoch (`train.epochs` is the number of rounds). After a worker failure it resumes from the last round's `model.ubj` through `modelport.latest_artifacts()`.
- `input_stats` keeps the default (the numeric input columns), so drift is measured per feature.
- `load(None)` gives an untrained model that predicts every class with probability 1/7, so `modelport validate` can run predict before any training.

Dataset: Covertype, Jock A. Blackard (1998), UCI Machine Learning Repository, CC BY 4.0. See `NOTICE`.
