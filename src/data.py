from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parents[1] / "data"  # чтобы работало откуда угодно
DAY_MS = 24 * 60 * 60 * 1000
CUT_EVENTS = 5  # столько событий организаторы вырезают перед целевым кликом


def load():
    """читаем с файлов данные"""
    train = pl.read_parquet(DATA / "train.parquet")
    test = pl.read_parquet(DATA / "test.parquet")
    ed = pl.read_parquet(DATA / "event_dictionary.parquet")
    return train, test, ed


def flatten(df, ed):
    """одна строка = один клиент со списком событий -> одна строка = одно событие + приклеиваем тип из справочника (ф-ция из EDA)"""
    return (
        df.select("example_id", "events")
        .explode("events")
        .unnest("events")
        .join(ed, on="encoded_event", how="left")
    )


def time_split(train, events, days=7):
    """тот же сплит, как и в validation, но дни передаем параметрами, на случай если захотим проверить другие сплиты"""
    ends = events.group_by("example_id").agg(pl.col("timestamp").max().alias("end"))
    cut = ends["end"].max() - days * DAY_MS
    val_part = ends.filter(pl.col("end") >= cut).select("example_id")
    tr = train.join(val_part, on="example_id", how="anti")
    val = train.join(val_part, on="example_id", how="semi")
    return tr, val, cut


def click_examples(events, products, min_history=30, per_example=None, seed=0):
    """Каждый клик 469 внутри истории -> новый пример, как у организаторов:
    история до клика минус 5 событий, ответ - продукт клика"""
    # через i нумеруем порядок действий клиента
    ev = events.with_columns(pl.int_range(pl.len()).over("example_id").alias("i"))
    clicks = ev.filter(
        (pl.col("encoded_event") == 469)  # это целевой клик
        & pl.col("encoded_product").is_in(products)  # по одному из наших 32 продуктов
        & (pl.col("i") >= min_history)  # перед ним достаточно истории
    ).select(
        "example_id",
        pl.col("i").alias("click_i"),  # где клик
        pl.col("timestamp").alias("click_ts"),  # когда клик
        pl.col("encoded_product").alias("target_product"),  # на что кликнули = ответ
    )
    # оставляем per_example случайных кликов на клиента если нужно
    # тк в трейне очень много кликов, нам все не нужны
    # из теста берем все возможные которые не ответы
    if per_example:
        clicks = (
            clicks.sample(fraction=1.0, shuffle=True, seed=seed)
            .group_by("example_id")
            .head(per_example)
        )

    # имя нового примера
    clicks = clicks.with_columns(
        (pl.col("example_id") + "_" + pl.col("click_i").cast(pl.Utf8)).alias("new_id")
    )
    new_events = (
        ev.join(clicks.select("example_id", "click_i", "new_id"), on="example_id")
        .filter(pl.col("i") < pl.col("click_i") - CUT_EVENTS)
        .drop("example_id", "i", "click_i")
        .rename({"new_id": "example_id"})
    )
    labels = clicks.select(
        pl.col("new_id").alias("example_id"), "target_product", "click_ts"
    )
    return new_events.select(events.columns), labels
