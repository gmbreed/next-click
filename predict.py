from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.data import load, flatten
from src.features import popularity, build_features
from src.model import predict_proba, debias, top1

ROOT = Path(__file__).resolve().parent
BETAS = [0.0, 0.25]  # сила поправки; для каждого значения свой файл сабмита


def main():
    # 1. данные
    train, test, ed = load()
    ftr, fte = flatten(train, ed), flatten(test, ed)
    products = sorted(train["target_product"].unique().to_list())

    # 2. признаки для test — та же функция, та же популярность, тот же фильтр живых
    pop = popularity([ftr, fte], products)
    cand = build_features(fte, products, pop).filter(pl.col("views_3d") > 0)

    # 3. все сохранённые модели (по одной на seed), усредняем их вероятности
    models = [
        lgb.Booster(model_file=str(f))
        for f in sorted((ROOT / "models").glob("ranker_*.txt"))
    ]
    features = models[0].feature_name()
    proba = predict_proba(models, cand, features, products)

    # 4. для каждого beta — свой файл сабмита
    (ROOT / "submissions").mkdir(exist_ok=True)
    for beta in BETAS:
        pred = top1(cand, debias(cand, proba, beta))
        sub = test.select("example_id").join(pred, on="example_id", how="left")
        sub = sub.rename({"pred": "encoded_product"})
        out = ROOT / "submissions" / f"ranker_beta{beta}.csv"
        sub.write_csv(out)
        print(out.name, sub.shape, "пропусков:", sub["encoded_product"].null_count())


if __name__ == "__main__":
    main()
