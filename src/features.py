import polars as pl

EVENT_CODES = [13, 399, 542, 835, 469, 162, 271, 759]  # самые частые события с продуктом, были в EDA
HOUR = 3600
DAY = 24 * HOUR

def candidates(events, products):
    """строим базовую табличку с маркером для клика и продукта у пользователя
    Для каждого примера - по строке на каждый продукт.
    Каждая строка - вопрос к модели: «кликнет ли клиент id на продукт num?
    end - точка отсчета: можем находить как давно у нас было событие относительно истории клиента"""
    ends = events.group_by("example_id").agg(pl.col("timestamp").max().alias("end"))
    prods = pl.DataFrame({"encoded_product": products}).cast(pl.Int32)
    return ends.join(prods, how="cross")


def add_target(cand, labels):
    """y = 1 у продукта, на который клиент кликнул, иначе 0."""
    return cand.join(
        labels.select("example_id", "target_product"), on="example_id"
    ).with_columns(
        (pl.col("encoded_product") == pl.col("target_product")).cast(pl.Int8).alias("y")
    )


def history_features(events):
    """Признаки пары «пример x продукт» по истории: сколько раз, как давно, какой по свежести"""
    ends = events.group_by("example_id").agg(pl.col("timestamp").max().alias("end"))
    ev = events.filter(pl.col("encoded_product").is_not_null()).join(
        ends, on="example_id"
    )
    ago = (pl.col("end") - pl.col("timestamp")) / 1000  # секунд до конца истории
    feats = ev.group_by("example_id", "encoded_product").agg(
        pl.len().alias("n_any"),  # сколько всего событий с продуктом
        # сколько раз было каждое событие из EVENT_CODES: n_13 — показы баннера, n_469 — целевые клики и т.д.
        *[(pl.col("encoded_event") == c).sum().alias(f"n_{c}") for c in EVENT_CODES],
        (pl.col("event_type") == "click").sum().alias("n_click"),  # клики любого вида
        (ago < HOUR).sum().alias("n_1h"),  # сколько событий с продуктом за последний час
        (ago < DAY).sum().alias("n_1d"),  # сколько событий с продуктом за последние сутки
        ago.min().alias("since_last"),  # сколько секунд назад было последнее событие с продуктом
        ago.max().alias("since_first"),  # сколько секунд назад было первое событие с продуктом
        ago.filter(pl.col("encoded_event") == 13).min().alias("since_13"),  # сколько секунд назад был последний показ продукта
        ago.filter(pl.col("event_type") == "click").min().alias("since_click"),  # сколько секунд назад был последний клик по продукту
    )
    # ранги свежести внутри клиента: 1 = самый свежий продукт (rk_13 == 1 — это правило view)
    # 2;3;4;... - чем больше ранг, тем более старое событие
    # ранги нужны для уравновешивания клиентов: один клиент может сделать 1 действие с продуктом за час,
    # а другой за то же время 10, это тоже надо учитывать, а не только время от конца
    return feats.with_columns(
        pl.col("since_13").rank("ordinal").over("example_id").alias("rk_13"), #попробуем добавить ранг только для самого частого действия в отдельности
        #later: попробовать для других действий тоже добавить ранг
        pl.col("since_last").rank("ordinal").over("example_id").alias("rk_any"),
        pl.col("since_click").rank("ordinal").over("example_id").alias("rk_click"),
    )


def add_features(cand, feats):
    """Приклеить признаки к кандидатам; где событий не было, счётчики = 0"""
    return cand.join(
        feats, on=["example_id", "encoded_product"], how="left"
    ).with_columns(
        pl.col("^n_.*$").fill_null(0)
    )  # все столбцы n_* заполняем нулями с помощью регулярки, время (since_*) и ранги (rk_*) оставляем null
