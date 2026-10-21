"""LightGBM with a small hyperparameter search on the pipeline's own validation split (training rows only).

The same search is applied to every text arm, baselines included, so comparisons stay fair. Grid (12 configs):
learning_rate {0.03, 0.1} x max_depth {-1 (num_leaves=31), 6} x colsample_bytree {0.3, 0.6, 1.0}; each config uses
early stopping (50 rounds, up to 2000 trees) on the validation rows; the config with the lowest validation loss is
kept as trained (no refit, so the validation rows never become training rows).
"""
import itertools

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame, Series

from multabench.baselines.lgbm import LightGBM
from multabench.constants import SEED

GRID = {"learning_rate": [0.03, 0.1], "max_depth": [-1, 6], "colsample_bytree": [0.3, 0.6, 1.0]}


class LightGBMTuned(LightGBM):
    MODEL_NAME = "LightGBM-tuned"
    SHORT_NAME = "lgbt"

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_cols = [c for c in x_train.columns if c in self.categorical_features]
        best = None
        for lr, depth, col in itertools.product(*GRID.values()):
            params = dict(n_estimators=2000, learning_rate=lr, max_depth=depth, colsample_bytree=col,
                          verbose=-1, random_state=SEED)
            m = (LGBMClassifier if self.is_cls else LGBMRegressor)(**params)
            m.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=cat_cols,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            loss = list(m.best_score_["valid_0"].values())[0]
            if best is None or loss < best[0]:
                best = (loss, m, {"learning_rate": lr, "max_depth": depth, "colsample_bytree": col,
                                  "n_trees": m.best_iteration_})
        self.model_ = best[1]
        self.best_params_ = best[2]
        print(f"🎛️ tuned LightGBM: {self.best_params_} (val loss {best[0]:.4f})")
