from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parents[1] / "data"  # чтобы работало откуда угодно
DAY_MS = 24 * 60 * 60 * 1000


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
