import lightgbm as lgb
import pandas as pd
import polars as pl
import numpy as np

NOT_FEATURES = ["example_id", "end", "day", "target_product", "y", "w"]
PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=0,
    verbose=-1,
)


def to_X(df, features, products):
    """polars -> pandas для LightGBM; номер продукта — категория."""
    X = df.select(features).to_pandas()
    X["encoded_product"] = pd.Categorical(X["encoded_product"], categories=products)
    return X


def fit_ranker(df, features, products, seed=0):
    """Обучить ранкер: внутри каждого клиента поднять наверх продукт с y = 1."""
    # выкидываем клиентов, у которых правильного продукта нет среди кандидатов
    # (после фильтра мёртвых продуктов, например 55 в феврале, когда его не показывали)
    has_target = (
        df.group_by("example_id").agg(pl.col("y").max()).filter(pl.col("y") == 1)
    )
    df = df.join(has_target.select("example_id"), on="example_id", how="semi")
    df = df.sort("example_id", "encoded_product")  # строки одного клиента должны идти подряд
    # для lgbm ranker нужен group — сколько строк подряд относятся к одному клиенту: [32, 32, 32, ...]
    groups = df.group_by("example_id", maintain_order=True).len()["len"].to_numpy()
    model = lgb.LGBMRanker(**{**PARAMS, "random_state": seed})
    weight = df["w"].to_numpy() if "w" in df.columns else None  # вес примера (апрельские важнее)
    model.fit(to_X(df, features, products), df["y"].to_numpy(), group=groups, sample_weight=weight)
    return model


def top1(df, scores):
    """Для каждого клиента — продукт с максимальным скором."""
    return (
        df.with_columns(pl.Series("score", scores))
        .sort("score", "encoded_product", descending=[True, False])
        .group_by("example_id")
        .first()
        .select("example_id", pl.col("encoded_product").alias("pred"))
    )


def accuracy(pred, labels):
    """Доля клиентов, у которых предсказание совпало с ответом (клиенты без предсказания — ошибка)."""
    return (
        labels.join(pred, on="example_id", how="left")
        .select((pl.col("pred") == pl.col("target_product")).fill_null(False).mean())
        .item()
    )


def proba_by_example(df, scores):
    """Скоры модели -> вероятности внутри клиента (softmax: сумма по его продуктам = 1)."""
    return (
        df.select("example_id")
        .with_columns(pl.Series("s", scores))
        .with_columns(
            (pl.col("s") - pl.col("s").max().over("example_id")).exp().alias("e")
        )
        .select(pl.col("e") / pl.col("e").sum().over("example_id"))
        .to_series()
        .to_numpy()
    )


def predict_proba(models, df, features, products):
    """Усредняем вероятности нескольких моделей (разные seed)."""
    X = to_X(df, features, products)
    return sum(proba_by_example(df, m.predict(X)) for m in models) / len(models)
