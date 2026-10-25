"""The CSV entry point: task inference, every task x learner end to end (fake E5, as in test_sae), output format,
and that test labels never influence predictions."""
import numpy as np
import pandas as pd
import pytest
import torch

from run_csv import fit_predict, infer_task, labels, main
from tests.test_sae import _toy_task, fake_e5, fake_tar  # noqa: F401  (fixtures)


def _frame(task, n=1500):
    x, y = _toy_task(task, n=n)
    df = x.copy()
    df["target"] = y.map({0: "no", 1: "yes"}) if task == "bin" else y
    return df.iloc[:1000].reset_index(drop=True), df.iloc[1000:].reset_index(drop=True)


def test_infer_task():
    assert infer_task(pd.Series(["a", "b", "a"])) == "binary"
    assert infer_task(pd.Series(["a", "b", "c"])) == "multiclass"
    assert infer_task(pd.Series([0, 1, 1, 0])) == "binary"
    assert infer_task(pd.Series([True, False])) == "binary"
    assert infer_task(pd.Series(np.linspace(0, 1, 50))) == "regression"
    assert infer_task(pd.Series(np.arange(100))) == "regression"
    with pytest.raises(ValueError, match="--task"):
        infer_task(pd.Series([0, 1, 2, 3, 2]))
    with pytest.raises(ValueError):
        infer_task(pd.Series([1, 1, None]))


def test_labels_unify_int_and_float():
    assert labels(pd.Series([1, 2])).tolist() == ["1", "2"]
    assert labels(pd.Series([1.0, 2.0, None])).tolist()[:2] == ["1", "2"]
    assert labels(pd.Series(["a", None])).isna().tolist() == [False, True]


@pytest.mark.parametrize("learner", ["lgbt", "cat"])
@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_fit_predict(fake_e5, fake_tar, task, learner):
    train, test = _frame(task)
    explicit = {"bin": "auto", "mul": "multiclass", "reg": "auto"}[task]   # integer 0-3 is ambiguous on purpose
    out, metric = fit_predict(train, test, "target", task=explicit, learner=learner, device=torch.device("cpu"))
    assert len(out) == len(test) and (out.index == test.index).all()
    if task == "reg":
        assert list(out.columns) == ["prediction"] and np.isfinite(out.prediction).all()
        assert metric["course_metric"] == "r2" and metric["course_score"] > 0.5
    else:
        probs = out.filter(like="proba_")
        expected = ["proba_no", "proba_yes"] if task == "bin" else [f"proba_{c}" for c in range(4)]
        assert list(probs.columns) == expected
        assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
        assert (out.predicted_label == probs.idxmax(axis=1).str[len("proba_"):]).all()
        assert metric["course_score"] > 0.7
    assert len(fake_tar) == 1                               # one fine-tune per text column set, on train rows only


def test_test_labels_do_not_change_predictions(fake_e5, fake_tar):
    train, test = _frame("bin")
    a, m = fit_predict(train, test, "target", device=torch.device("cpu"))
    b, _ = fit_predict(train, test.drop(columns=["target"]), "target", device=torch.device("cpu"))
    pd.testing.assert_frame_equal(a, b)
    assert m is not None


def test_cli_with_holdout(fake_e5, fake_tar, tmp_path, monkeypatch):
    import run_csv
    train, test = _frame("mul")
    df = pd.concat([train, test], ignore_index=True)
    df.insert(0, "row_id", np.arange(len(df)))
    df.loc[5, "target"] = None                               # rows without a target are dropped
    df.to_csv(tmp_path / "data.csv", index=False)
    monkeypatch.setattr(run_csv, "fit_predict", lambda *a, **k: fit_predict(*a, **k, device=torch.device("cpu")))
    main(["--train", str(tmp_path / "data.csv"), "--target", "target", "--task", "multiclass", "--test-size", "0.2",
          "--drop-cols", "row_id", "--out", str(tmp_path / "pred.csv")])
    pred = pd.read_csv(tmp_path / "pred.csv")
    assert len(pred) == round(0.2 * (len(df) - 1)) and "predicted_label" in pred and "row_id" not in pred


def test_cli_compare_shares_one_finetune(fake_e5, fake_tar, tmp_path, monkeypatch, capsys):
    import run_csv
    train, test = _frame("reg")
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    monkeypatch.setattr(run_csv, "fit_predict", lambda *a, **k: fit_predict(*a, **k, device=torch.device("cpu")))
    main(["--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"), "--target", "target",
          "--compare", "--out", str(tmp_path / "pred.csv")])
    printed = capsys.readouterr().out
    assert "e5-small-tar (Joint TAR baseline): r2" in printed and "e5-small-tar-fstack (" in printed
    assert len(fake_tar) == 1                                 # the baseline and the method share one fine-tune
