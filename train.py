from pathlib import Path

import polars as pl

from src.data import load, flatten, time_split, click_examples
from src.features import popularity, fresh_popularity, build_features, add_target
from src.features import tail_codes, client_folds, context_tables, add_context
from src.model import NOT_FEATURES, fit_ranker, predict_proba, top1, accuracy

SEEDS = [0, 1, 2]  # несколько моделей с разным seed, усредняем
T0 = 29592834 * 60000  # 07.04 13:54 UTC — банк сменил режим показов
W_APRIL = 5.0  # вес примеров с кликом после конца train
W_POST = 20.0  # вес примеров с кликом после смены режима

ROOT = Path(__file__).resolve().parent
ALIVE = (
    pl.col("views_3d") > 0
)  # кандидаты — только продукты, которые показывали последние 3 дня


def make_candidates(events, labels, products, pop, fresh):
    """Кандидаты + признаки + ответ, только живые продукты."""
    return add_target(build_features(events, products, pop, fresh), labels).filter(ALIVE)


def main():
    # 1. данные
    train, test, ed = load()
    ftr, fte = flatten(train, ed), flatten(test, ed)
    products = sorted(train["target_product"].unique().to_list())
    pop = popularity([ftr, fte], products)
    fresh = fresh_popularity([ftr, fte], products)

    # 2. кандидаты для настоящих примеров train
    cand = make_candidates(ftr, train, products, pop, fresh)

    # 3. новые примеры из кликов внутри историй (train — по 1 на клиента, test — все)
    ev_tr, lab_tr = click_examples(ftr, products, per_example=1)
    ev_te, lab_te = click_examples(fte, products)
    extra = pl.concat(
        [
            make_candidates(ev_tr, lab_tr, products, pop, fresh),
            make_candidates(ev_te, lab_te, products, pop, fresh),
        ]
    )
    extra_labels = pl.concat([lab_tr, lab_te])

    # контекст экрана: коды последних событий (все 1076 кодов) -> какой продукт после них кликают.
    # таблицы считаем по примерам train; каждому клиенту — таблица без его собственных ответов
    clients = pl.concat([train.select("example_id", "client_id"), test.select("example_id", "client_id")])
    t_tr, t_ex = tail_codes(ftr), pl.concat([tail_codes(ev_tr), tail_codes(ev_te)])
    folds = client_folds(pl.concat([t_tr, t_ex])["example_id"].unique(), clients)
    pool = pl.concat([train.select("example_id", "target_product"), lab_tr.select("example_id", "target_product")])
    tables = context_tables(pl.concat([t_tr, tail_codes(ev_tr)]), pool, folds, products)
    cand = add_context(cand, t_tr, folds, tables)
    extra = add_context(extra, t_ex, folds, tables)
    features = [c for c in cand.columns if c not in NOT_FEATURES]
    (ROOT / "models").mkdir(exist_ok=True)
    tables.write_parquet(ROOT / "models" / "context.parquet")  # predict.py возьмёт те же таблицы

    # вес примера: чем ближе клик к режиму test, тем важнее
    train_end = ftr["timestamp"].max()
    cand = cand.with_columns(pl.lit(1.0).alias("w"))
    extra = (
        extra.join(extra_labels.select("example_id", "click_ts"), on="example_id")
        .with_columns(
            pl.when(pl.col("click_ts") >= T0).then(W_POST)
            .when(pl.col("click_ts") > train_end).then(W_APRIL)
            .otherwise(1.0)
            .alias("w")
        )
        .drop("click_ts")
    )

    # 4. сплит по времени: val = последняя неделя train
    #    новые примеры в обучение — только с кликом ДО cut, иначе подсмотрим будущее
    tr_ex, val_ex, cut = time_split(train, ftr)
    tr = cand.filter(pl.col("end") < cut)
    val = cand.filter(pl.col("end") >= cut)
    extra_tr = extra.join(
        extra_labels.filter(pl.col("click_ts") < cut).select("example_id"),
        on="example_id",
        how="semi",
    )

    # 5. две валидации
    #    testlike — val без ответов-продуктов, которых нет в апреле
    #    april — новые примеры из кликов в test-историях после конца train
    april_products = fte.filter(
        (pl.col("encoded_event") == 13) & (pl.col("timestamp") > train_end)
    )["encoded_product"].unique()
    val_labels = val_ex.select("example_id", "target_product")
    val_testlike = val_labels.filter(
        pl.col("target_product").is_in(april_products.implode())
    )
    april_labels = lab_te.filter(pl.col("click_ts") > train_end).select(
        "example_id", "target_product"
    )
    april = extra.join(april_labels.select("example_id"), on="example_id", how="semi")

    # 6. модели (по одной на seed) и сравнение с правилом view
    models = [fit_ranker(pl.concat([tr, extra_tr]), features, products, seed=s) for s in SEEDS]
    print(len(features), "признаков,", extra_tr["example_id"].n_unique(), "доп. примеров,", april_labels.height, "апрельских")
    p_val = top1(val, -val["rk_13"].fill_null(999).to_numpy())
    p_apr = top1(april, -april["rk_13"].fill_null(999).to_numpy())
    print(f"view    testlike {accuracy(p_val, val_testlike):.4f}  april {accuracy(p_apr, april_labels):.4f}")
    p_val = top1(val, predict_proba(models, val, features, products))
    p_apr = top1(april, predict_proba(models, april, features, products))
    print(f"ranker  testlike {accuracy(p_val, val_testlike):.4f}  april {accuracy(p_apr, april_labels):.4f}")

    # 7. финальные модели: всё, что есть (train + все новые примеры)
    #    на каждый seed две модели — с весами и без; predict.py усредняет все шесть
    (ROOT / "models").mkdir(exist_ok=True)
    full = pl.concat([cand, extra])
    for s in SEEDS:
        with_w = fit_ranker(full, features, products, seed=s)
        with_w.booster_.save_model(str(ROOT / "models" / f"ranker_w_{s}.txt"))
        no_w = fit_ranker(full.drop("w"), features, products, seed=s)
        no_w.booster_.save_model(str(ROOT / "models" / f"ranker_{s}.txt"))
    print("модели сохранены в models/ranker_*.txt")


if __name__ == "__main__":
    main()
