"""트리·표 모델 6종 시험(합성 자료) — 범주 변환 train/predict 같음·없던 범주·n_iter 예측·기본 설정·씨앗 재현·
로지스틱 원핫·표준화 통계·반복 수 고르기(§3)·CatBoost 학습률 읽기(부록 A-12)·저장."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from lst import metrics as mt
from lst import models_tree as mtree

CATS = ["sido", "p3"]


def make_data(n=1500, seed=0, n_cat=12):
    rng = np.random.default_rng(seed)
    sido = rng.choice([f"{i:02d}" for i in range(11, 11 + n_cat)], n)
    p3 = rng.choice(["B0204000000000", "C0310020000000", "코드없음", "A0101000000000"], n)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(loc=5, scale=3, size=n).astype(np.float32)
    x2[rng.random(n) < 0.15] = np.nan
    h = rng.integers(1, 61, n).astype(np.float32)
    logit = 1.2 * x1 + (sido == "11") * 1.5 + (p3 == "코드없음") * 1.0 - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    X = pd.DataFrame({"sido": sido, "x1": x1, "p3": p3, "x2": x2, "h": h})
    return X, y


def fit(algo, X, y, seed=42, n_jobs=2, **kw):
    return mtree.make_model(algo, cat_cols=CATS, **kw).fit(X, y, seed=seed, n_jobs=n_jobs)


@pytest.fixture(scope="module")
def data():
    return make_data()


# ---------------------------------------------------------------- 범주 변환
@pytest.mark.parametrize("algo", mtree.ALGOS)
def test_category_encoding_train_values_and_consistent(algo, data):
    X, y = data
    m = fit(algo, X, y)
    for c in CATS:
        assert m.categories_[c] == sorted(set(X[c]))
    # 같은 행을 순서만 바꿔 예측 = 예측도 같은 순서로 바뀜(행마다 같은 변환)
    perm = np.random.default_rng(1).permutation(len(X))
    p = m.predict_proba(X)
    p2 = m.predict_proba(X.iloc[perm].reset_index(drop=True))
    assert p.dtype == np.float64 and p.shape == (len(X),)
    np.testing.assert_array_equal(p[perm], p2)
    # 다른 자료(범주 일부만)로 예측해도 범주 목록은 train 것 그대로
    sub = X[X["sido"].isin(["11", "12"])].reset_index(drop=True)
    e_sub = m._encode(sub)
    if algo in ("XGB", "LGBM"):
        assert list(e_sub["sido"].cat.categories) == m.categories_["sido"]
        e_full = m._encode(X)
        np.testing.assert_array_equal(e_sub["sido"].cat.codes.to_numpy(),
                                      e_full["sido"].cat.codes.to_numpy()[X["sido"].isin(["11", "12"]).to_numpy()])


def test_rf_codes_sorted_order(data):
    X, y = data
    m = fit("RF", X, y)
    enc = m._encode(X)
    j = m.columns_.index("p3")
    order = sorted(set(X["p3"]))
    expect = np.array([order.index(v) for v in X["p3"]], dtype=np.float32)
    np.testing.assert_array_equal(enc[:, j], expect)
    assert enc.dtype == np.float32 and isinstance(enc, np.ndarray)


@pytest.mark.parametrize("algo", mtree.ALGOS)
def test_unseen_category(algo, data):
    X, y = data
    m = fit(algo, X, y)
    Xu = X.head(50).copy()
    Xu["sido"] = "99"  # train 에 없던 값
    p = m.predict_proba(Xu)
    assert np.isfinite(p).all()
    enc = m._encode(Xu)
    if algo in ("XGB", "LGBM"):
        assert enc["sido"].isna().all()
        # 없던 값 = 결측과 같은 예측
        Xn = X.head(50).copy()
        Xn_enc = m._encode(Xn)
        Xn_enc["sido"] = pd.Categorical([np.nan] * 50, categories=m.categories_["sido"])
        np.testing.assert_array_equal(m._predict(Xn_enc, None).astype(np.float64), p)
    elif algo in ("RF", "ET"):
        assert np.isnan(enc[:, m.columns_.index("sido")]).all()
    elif algo == "LR":
        names = m.design_columns_
        idx = [i for i, nm in enumerate(names) if nm.startswith("sido==")]
        assert (enc[:, idx] == 0).all()


def test_category_column_rules(data):
    X, y = data
    Xb = X.copy()
    Xb.loc[3, "sido"] = None
    with pytest.raises(ValueError, match="결측"):
        fit("XGB", Xb, y)
    Xi = X.copy()
    Xi["sido"] = Xi["sido"].astype(object)
    Xi.loc[3, "sido"] = 11
    with pytest.raises(TypeError):
        fit("CAT", Xi, y)
    m = fit("LGBM", X, y)
    with pytest.raises(ValueError, match="칼럼"):
        m.predict_proba(X[["x1", "sido", "p3", "x2", "h"]])
    with pytest.raises(ValueError, match="eval_set"):
        mtree.make_model("XGB", cat_cols=CATS).fit(X, y, seed=42, n_jobs=1, eval_set=[(X, y)])
    with pytest.raises(ValueError):
        mtree.make_model("XGB", cat_cols=CATS).fit(X, y, seed=42, n_jobs=9)


# ---------------------------------------------------------------- n_iter
@pytest.mark.parametrize("algo", ["XGB", "LGBM"])
def test_n_iter_equals_model_with_that_many_trees(algo, data):
    X, y = data
    big = fit(algo, X, y, max_iter=300)
    for k in (10, 120, 300):
        small = fit(algo, X, y, max_iter=k)
        np.testing.assert_array_equal(big.predict_proba(X, n_iter=k), small.predict_proba(X))


def test_n_iter_catboost(data):
    """CatBoost: n_iter = 라이브러리 ntree_end 와 같음. iterations 로 바뀌는 자동값(leaf_estimation_iterations)이
    같은 두 모델(300·200, 학습률 고정)이면 'k 트리 모델'과도 같음."""
    X, y = data
    params = {"learning_rate": 0.05}
    big = fit("CAT", X, y, max_iter=300, params=params)
    small = fit("CAT", X, y, max_iter=200, params=params)
    np.testing.assert_array_equal(big.predict_proba(X, n_iter=200), small.predict_proba(X))
    np.testing.assert_array_equal(big.predict_proba(X, n_iter=50),
                                  big.estimator_.predict_proba(X, ntree_end=50)[:, 1])
    a = big.config_dump()["applied_params"]["get_all_params"]
    b = small.config_dump()["applied_params"]["get_all_params"]
    assert a["leaf_estimation_iterations"] == b["leaf_estimation_iterations"]


@pytest.mark.parametrize("algo", mtree.BOOSTING)
def test_predict_multi_matches_single(algo, data):
    X, y = data
    m = fit(algo, X, y, max_iter=250)
    pts = mtree.checkpoints(250)
    multi = m.predict_proba_multi(X, pts)
    assert multi.shape == (len(X), len(pts)) and multi.dtype == np.float64
    for j, k in enumerate(pts):
        np.testing.assert_array_equal(multi[:, j], m.predict_proba(X, n_iter=k))
    with pytest.raises(ValueError):
        m.predict_proba(X, n_iter=251)


@pytest.mark.parametrize("algo", ["RF", "ET", "LR"])
def test_no_n_iter_for_non_boosting(algo, data):
    X, y = data
    m = fit(algo, X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X, n_iter=10)
    with pytest.raises(ValueError):
        mtree.make_model(algo, cat_cols=CATS, max_iter=100)


# ---------------------------------------------------------------- 기본 설정 그대로
EXPECTED_NON_DEFAULT = {
    "XGB": {"random_state", "n_jobs"},
    "LGBM": {"random_state", "n_jobs", "verbose"},
    "CAT": {"random_seed", "thread_count", "verbose", "allow_writing_files"},
    "RF": {"random_state", "n_jobs"},
    "ET": {"random_state", "n_jobs"},
    "LR": {"random_state", "max_iter"},
}


@pytest.mark.parametrize("algo", mtree.ALGOS)
def test_default_config(algo, data):
    X, y = data
    m = fit(algo, X, y)
    nd = m.non_default_params()
    assert set(nd) == EXPECTED_NON_DEFAULT[algo], nd
    d = m.config_dump()
    json.dumps(d, ensure_ascii=False)  # JSON 가능
    assert d["columns"] == list(X.columns) and d["seed"] == 42 and d["n_jobs"] == 2
    ap = d["applied_params"]
    if algo == "XGB":
        lp = ap["save_config"]["learner"]
        assert lp["gradient_booster"]["tree_train_param"]["eta"].startswith("0.3")  # 기본 학습률
        assert d["fit_info"]["n_trees_built"] == 100                                  # 기본 반복 수
        assert ap["get_params"]["missing"] is None                                    # NaN(기본) → JSON null
        assert ap["get_params"]["enable_categorical"] is True
    elif algo == "LGBM":
        assert ap["booster_params"]["num_iterations"] == 100
        assert ap["get_params"]["learning_rate"] == 0.1
        assert ap["booster_params"].get("zero_as_missing", False) is False
        assert ap["booster_params"]["categorical_column"] == [0, 2]
    elif algo == "CAT":
        allp = ap["get_all_params"]
        assert allp["iterations"] == 1000 and allp["nan_mode"] == "Min" and allp["use_best_model"] is False
        assert d["fit_info"]["learning_rate_applied"] == allp["learning_rate"]  # 자동 학습률
    elif algo == "LR":
        assert d["fit_info"]["n_iter_"] >= 1 and "converged" in d["fit_info"]


def test_chain_params_and_whitelist(data):
    X, y = data
    m = fit("CAT", X, y, max_iter=200, params={"learning_rate": 0.03})
    assert set(m.non_default_params()) == EXPECTED_NON_DEFAULT["CAT"] | {"iterations", "learning_rate"}
    m = fit("LGBM", X, y, max_iter=200, params={"lambda_l2": 2.0, "bagging_fraction": 0.8, "bagging_freq": 1})
    gp = m.estimator_.get_params()
    assert gp["reg_lambda"] == 2.0 and gp["subsample"] == 0.8 and gp["subsample_freq"] == 1
    with pytest.raises(ValueError, match="A-7"):
        mtree.make_model("XGB", cat_cols=CATS, params={"gamma": 1.0})
    with pytest.raises(ValueError):
        mtree.make_model("CAT", cat_cols=CATS, params={"loss_function": "CrossEntropy"})
    mtree.make_model("CAT", cat_cols=CATS, params={"loss_function": "Focal:focal_alpha=0.75;focal_gamma=2"})
    m = fit("RF", X, y, params={"class_weight": {0: 1, 1: 5.0}})
    assert m.estimator_.get_params()["class_weight"] == {0: 1, 1: 5.0}


# ---------------------------------------------------------------- 씨앗 재현
@pytest.mark.parametrize("algo", mtree.ALGOS)
def test_seed_reproducible(algo, data):
    X, y = data
    a = fit(algo, X, y, seed=42).predict_proba(X)
    b = fit(algo, X, y, seed=42).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    if algo in ("RF", "ET", "CAT"):
        c = fit(algo, X, y, seed=7).predict_proba(X)
        assert not np.array_equal(a, c)


# ---------------------------------------------------------------- 로지스틱
def test_logistic_onehot_and_standardize_train_stats(data):
    X, y = data
    Xi = X.copy()
    Xi["x2__na"] = Xi["x2"].isna().astype(np.float32)
    m = mtree.make_model("LR", cat_cols=CATS, indicator_cols=["x2__na"]).fit(Xi, y, seed=42, n_jobs=1)
    for c in ("x1", "x2", "h"):
        v = Xi[c].to_numpy(dtype=np.float64)
        v = v[~np.isnan(v)]
        assert m.mean_[c] == pytest.approx(v.mean(), rel=0, abs=1e-12)
        assert m.scale_[c] == pytest.approx(v.std(ddof=0), rel=0, abs=1e-12)
    assert "x2__na" not in m.mean_  # 결측 표시는 표준화 안 함
    # 다른 자료로 변환해도 train 통계를 씀
    Xt, _ = make_data(n=300, seed=5)
    Xt["x2__na"] = Xt["x2"].isna().astype(np.float32)
    Xt["sido"] = Xt["sido"].where(Xt.index % 7 != 0, "99")
    enc = m._encode(Xt)
    names = m.design_columns_
    j = names.index("x2")
    v = Xt["x2"].to_numpy(dtype=np.float64)
    expect = np.where(np.isnan(v), 0.0, (v - m.mean_["x2"]) / m.scale_["x2"])
    np.testing.assert_allclose(enc[:, j], expect, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(enc[:, names.index("x2__na")], Xt["x2__na"].to_numpy(dtype=np.float64))
    for c in CATS:
        idx = [i for i, nm in enumerate(names) if nm.startswith(c + "==")]
        assert [names[i].split("==", 1)[1] for i in idx] == sorted(set(X[c]))
        block = enc[:, idx]
        assert set(np.unique(block)) <= {0.0, 1.0}
        known = Xt[c].isin(m.categories_[c]).to_numpy()
        np.testing.assert_array_equal(block.sum(axis=1), known.astype(float))
    # 칼럼 순서 = X 칼럼 순서(범주 자리에 원핫 묶음)
    assert names[0].startswith("sido==") and names[len(m.categories_["sido"])] == "x1"
    p = m.predict_proba(Xt)
    np.testing.assert_allclose(p, m.estimator_.predict_proba(enc)[:, 1], rtol=0, atol=0)


def test_logistic_constant_and_all_nan_columns(data):
    X, y = data
    Xc = X.copy()
    Xc["const"] = np.float32(3.0)
    Xc["allnan"] = np.float32(np.nan)
    m = mtree.make_model("LR", cat_cols=CATS).fit(Xc, y, seed=42, n_jobs=1)
    assert m.scale_["const"] == 1.0 and m.all_nan_cols_ == ["allnan"]
    enc = m._encode(Xc)
    assert (enc[:, m.design_columns_.index("allnan")] == 0).all()
    assert (enc[:, m.design_columns_.index("const")] == 0).all()
    bad = Xc.copy()
    bad["x1"] = bad["x1"].replace({bad["x1"].iloc[0]: np.inf})
    with pytest.raises(ValueError, match="inf"):
        m.predict_proba(bad)


# ---------------------------------------------------------------- 반복 수 고르기(§3)
def test_checkpoints():
    pts = mtree.checkpoints(2000)
    assert len(pts) == 56 and pts[:3] == [10, 20, 30] and pts[19] == 200 and pts[20] == 250 and pts[-1] == 2000
    assert len(mtree.checkpoints(4000)) == 96
    assert mtree.next_max_iter(2000) == 4000 and mtree.next_max_iter(8000) == 16000
    assert mtree.next_max_iter(16000) is None
    with pytest.raises(ValueError):
        mtree.checkpoints(2010)


def test_select_n_iter_cross_common_ties():
    pts = [10, 20, 30, 40]
    curves = {
        "fold1": [0.10, 0.20, 0.30, 0.10],
        "fold2": [0.10, 0.30, 0.20, 0.10],
        "fold3": [0.10, 0.30, 0.20, 0.300004],  # 4자리 반올림 → 0.3000 = 20 과 같음 → 작은 N
    }
    r = mtree.select_n_iter(curves, pts)
    # fold1 교차 = fold2·3 평균 [0.1, 0.3, 0.2, 0.200002] → 20
    assert r["cross_n"]["fold1"] == 20
    # fold2 교차 = fold1·3 평균 [0.1, 0.25, 0.25, 0.200002] → 20·30 동점 → 20
    assert r["cross_n"]["fold2"] == 20
    # fold3 교차 = fold1·2 평균 [0.1, 0.25, 0.25, 0.1] → 20
    assert r["cross_n"]["fold3"] == 20
    assert r["fold_score"] == {"fold1": 0.20, "fold2": 0.30, "fold3": 0.30}
    assert r["common_n"] == 20  # 평균 [0.1, 0.2667, 0.2333, 0.1667]
    assert r["own_best_n"] == {"fold1": 30, "fold2": 20, "fold3": 20}
    assert r["any_at_cap"] is False
    assert r["cv_mean"] == pytest.approx((0.2 + 0.3 + 0.3) / 3)
    curves2 = {f: [0.1, 0.2, 0.3, 0.4] for f in ("fold1", "fold2", "fold3")}
    r2 = mtree.select_n_iter(curves2, pts)
    assert r2["common_n_at_cap"] and r2["any_at_cap"] and all(r2["own_best_at_last"].values())
    with pytest.raises(ValueError, match="NaN"):
        mtree.select_n_iter({**curves2, "fold1": [0.1, math.nan, 0.3, 0.4]}, pts)
    with pytest.raises(ValueError):
        mtree.select_n_iter({"fold1": [0.1] * 4, "fold2": [0.1] * 4}, pts)


def test_fold_curve_uses_given_score_fn(data):
    X, y = data
    m = fit("XGB", X, y, max_iter=250)
    pts = mtree.checkpoints(250)
    h = np.random.default_rng(3).integers(1, 61, len(X))
    u = np.random.default_rng(4).random(len(X))
    multi = m.predict_proba_multi(X, pts)
    curve = mtree.fold_curve(multi, y, h, u, mt.pr_auc_band_weighted)
    assert curve.shape == (len(pts),)
    assert curve[5] == mt.pr_auc_band_weighted(y, m.predict_proba(X, n_iter=pts[5]), h, u)


# ---------------------------------------------------------------- CatBoost 학습률 읽기
def test_catboost_read_auto_learning_rate(data):
    X, y = data
    m = mtree.make_model("CAT", cat_cols=CATS, max_iter=2000)
    lr = m.read_auto_learning_rate(X, y, seed=42, n_jobs=2)
    assert m.estimator_ is None  # 객체는 그대로
    full = mtree.make_model("CAT", cat_cols=CATS, max_iter=2000).fit(X, y, seed=42, n_jobs=2)
    assert lr == full.fit_info_["learning_rate_applied"]  # 2,000 반복 학습의 자동값과 같음
    assert mtree.mean_learning_rate([lr, lr * 2]) == pytest.approx(lr * 1.5)
    with pytest.raises(ValueError):
        mtree.make_model("CAT", cat_cols=CATS, params={"learning_rate": 0.1}).read_auto_learning_rate(
            X, y, seed=42, n_jobs=1)


# ---------------------------------------------------------------- 저장·n_jobs 사다리
@pytest.mark.parametrize("algo", mtree.ALGOS)
def test_save_load_roundtrip(algo, data, tmp_path):
    X, y = data
    m = fit(algo, X, y)
    path = m.save(tmp_path / f"{algo}.pkl")
    with pytest.raises(FileExistsError):
        m.save(path)
    m2 = mtree.TreeModel.load(path)
    assert type(m2) is type(m)
    np.testing.assert_array_equal(m.predict_proba(X), m2.predict_proba(X))


def test_retry_ladder():
    assert mtree.retry_n_jobs_ladder("RF", 8) == (8, 4, 2, 1)
    assert mtree.retry_n_jobs_ladder("ET", 6) == (6, 3, 1)
    assert mtree.retry_n_jobs_ladder("CAT", 8) == (8,)
    assert mtree.retry_n_jobs_ladder("XGB", 8) == (8,)
