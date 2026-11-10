from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.data import load, flatten
from src.features import popularity, build_features
from src.model import to_X, top1

ROOT = Path(__file__).resolve().parent


def main():
    train, test, ed = load()
    ftr, fte = flatten(train, ed), flatten(test, ed)
    products = sorted(train["target_product"].unique().to_list())

    # признаки для test — та же функция, та же популярность
    pop = popularity([ftr, fte], products)
    cand = build_features(fte, products, pop)
    cand = cand.filter(pl.col("views_3d") > 0)

    # загружаем модель и берём ровно те признаки, на которых она училась
    model = lgb.Booster(model_file=str(ROOT / "models" / "ranker.txt"))
    pred = top1(cand, model.predict(to_X(cand, model.feature_name(), products)))

    # в формате sample_submission: example_id, encoded_product
    sub = test.select("example_id").join(pred, on="example_id", how="left")
    sub = sub.rename({"pred": "encoded_product"})
    (ROOT / "submissions").mkdir(exist_ok=True)
    sub.write_csv(ROOT / "submissions" / "ranker.csv")
    print(sub.shape, "пропусков:", sub["encoded_product"].null_count())


if __name__ == "__main__":
    main()
