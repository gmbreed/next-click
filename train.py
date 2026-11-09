import polars as pl

from src.data import load, flatten, time_split
from src.features import popularity, build_features, add_target
from src.model import NOT_FEATURES, to_X, fit_ranker, top1, accuracy


def main():
    # 1. данные
    train, test, ed = load()
    ftr, fte = flatten(train, ed), flatten(test, ed)
    products = sorted(train["target_product"].unique().to_list())

    # 2. признаки
    pop = popularity([ftr, fte], products)
    cand = add_target(build_features(ftr, products, pop), train)
    features = [c for c in cand.columns if c not in NOT_FEATURES]

    # 3. сплит по времени: val = последняя неделя train
    tr_ex, val_ex, cut = time_split(train, ftr)
    tr = cand.filter(pl.col("end") < cut)
    val = cand.filter(pl.col("end") >= cut)

    # 4. test-подобная валидация: только ответы-продукты, которые показывают в апреле
    train_end = ftr["timestamp"].max()
    april = fte.filter(
        (pl.col("encoded_event") == 13) & (pl.col("timestamp") > train_end)
    )["encoded_product"].unique()
    val_labels = val_ex.select("example_id", "target_product")
    val_testlike = val_labels.filter(pl.col("target_product").is_in(april.implode()))

    # 5. модель и сравнение с правилом view
    model = fit_ranker(tr, features, products)
    pred = top1(val, model.predict(to_X(val, features, products)))
    view = top1(val, -val["rk_13"].fill_null(999).to_numpy())

    print(len(features), "признаков")
    for name, p in [("view", view), ("ranker", pred)]:
        print(
            f"{name:7s} val {accuracy(p, val_labels):.4f}  testlike {accuracy(p, val_testlike):.4f}"
        )


if __name__ == "__main__":
    main()
