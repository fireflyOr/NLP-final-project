"""Late fusion: a tabular-only LightGBM and a text-only model, combined by a linear meta-learner.

    tabular model : LightGBM (default settings, as in the benchmark) on the non-text columns only
    text model    : MLP on the concatenated frozen E5 embeddings of all text columns (the stacking text model)
    meta-learner  : logistic regression (classification) / linear regression (regression) on the two models'
                    out-of-fold outputs (log-probabilities / predictions)

Leakage control: the meta-learner is fitted on out-of-fold outputs of both base models (5 folds inside the training
rows; rows with identical text kept together). At prediction time the tabular model is refitted on all training rows
and the text model is the average of its fold models. The text model uses the target, so the bar is Joint TAR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression, LogisticRegression

from multabench.baselines.lgbm import LightGBM
from multabench.constants import SEED
from multabench.stack.text_stack import StackEncoder

N_SPLITS = 5


class LateFusion(LightGBM):
    MODEL_NAME = "LateFusion"
    SHORT_NAME = "late"
    USE_VAL_SPLIT = False          # all training rows go to the cross-fitted base models
    USE_TEXT_EMBEDDINGS = False    # text columns stay raw strings; only the text model reads them

    def fit(self, x: DataFrame, y: Series):
        self.fit_preprocessor(x_train=x, y_train=y)
        xp, yp = self.transform_preprocessor(x=x, y=y)
        self.text_cols = sorted(self.text_features)
        tab = self._tab(xp)
        yy = np.asarray(yp)
        self.text_model = StackEncoder(kind="frozen", text_cols=self.text_cols, is_cls=self.is_cls, device=self.device,
                                       n_splits=N_SPLITS).fit(xp, pd.Series(yy, index=xp.index))
        text_oof = self.text_model.transform(xp).to_numpy()          # out-of-fold for these rows
        folds = self.text_model.folds_                               # the same row folds (text duplicates together)
        self.has_tab = tab.shape[1] > 0
        tab_oof = np.zeros((len(xp), self._n_out() if self.has_tab else 0))
        if self.has_tab:
            for tr, ho in folds:
                m = self._tab_model().fit(tab.iloc[tr], yy[tr], categorical_feature=self._cats(tab))
                tab_oof[ho] = self._tab_out(m, tab.iloc[ho])
            self.tab_model = self._tab_model().fit(tab, yy, categorical_feature=self._cats(tab))
        self.meta = (LogisticRegression(max_iter=2000) if self.is_cls else LinearRegression()).fit(
            np.hstack([tab_oof, text_oof]), yy)

    def predict(self, x: DataFrame) -> np.ndarray:
        xp, _ = self.transform_preprocessor(x=x, y=None)
        text = self.text_model.transform(xp).to_numpy()
        z = np.hstack([self._tab_out(self.tab_model, self._tab(xp)), text]) if self.has_tab else text
        if not self.is_cls:
            return self.meta.predict(z)
        p = self.meta.predict_proba(z)
        full = np.zeros((len(z), self.d_output))
        full[:, self.meta.classes_.astype(int)] = p
        return full[:, 1] if self.d_output == 2 else full

    # ---------------- helpers
    def _tab(self, xp: DataFrame) -> DataFrame:
        return xp[[c for c in xp.columns if c not in self.text_cols]]

    def _cats(self, tab: DataFrame):
        cats = [c for c in tab.columns if c in self.categorical_features]
        return cats or "auto"

    def _tab_model(self):
        return (LGBMClassifier if self.is_cls else LGBMRegressor)(verbose=-1, random_state=SEED)

    def _n_out(self) -> int:
        return 1 if (not self.is_cls or self.d_output == 2) else self.d_output

    def _tab_out(self, m, tab: DataFrame) -> np.ndarray:
        if not self.is_cls:
            return m.predict(tab)[:, None]
        p = np.full((len(tab), self.d_output), 1e-6)
        p[:, m.classes_.astype(int)] = np.clip(m.predict_proba(tab), 1e-6, 1)
        p /= p.sum(axis=1, keepdims=True)
        if self.d_output == 2:
            return np.log(p[:, 1:2] / p[:, 0:1])
        return np.log(p)
