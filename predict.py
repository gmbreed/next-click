from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.data import load, flatten
from src.features import popularity, fresh_popularity, build_features
from src.features import tail_codes, client_folds, add_context
from src.model import predict_proba, top1

ROOT = Path(__file__).resolve().parent


def main():
    # 1. данные
    train, test, ed = load()
    ftr, fte = flatten(train, ed), flatten(test, ed)
    products = sorted(train["target_product"].unique().to_list())

    # 2. признаки для test — та же функция, та же популярность, тот же фильтр живых
    pop = popularity([ftr, fte], products)
    fresh = fresh_popularity([ftr, fte], products)
    cand = build_features(fte, products, pop, fresh).filter(pl.col("views_3d") > 0)
    tails = tail_codes(fte)
    folds = client_folds(test["example_id"], test.select("example_id", "client_id"))
    cand = add_context(cand, tails, folds, pl.read_parquet(ROOT / "models" / "context.parquet"))

    # 3–4. два сабмита: только модели с весами нового режима и смесь всех шести
    (ROOT / "submissions").mkdir(exist_ok=True)
    for name, pattern in [("ranker_w", "ranker_w_*.txt"), ("ranker_mix", "ranker*.txt")]:
        models = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "models").glob(pattern))]
        features = models[0].feature_name()
        pred = top1(cand, predict_proba(models, cand, features, products))
        sub = test.select("example_id").join(pred, on="example_id", how="left")
        sub = sub.rename({"pred": "encoded_product"})
        out = ROOT / "submissions" / f"{name}.csv"
        sub.write_csv(out)
        print(out.name, len(models), "моделей", sub.shape, "пропусков:", sub["encoded_product"].null_count())


if __name__ == "__main__":
    main()
