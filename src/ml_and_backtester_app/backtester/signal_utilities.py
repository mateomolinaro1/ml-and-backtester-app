import pandas as pd

class Momentum:

    @staticmethod
    def rolling_momentum(
            df: pd.DataFrame,
            nb_period: int,
            nb_period_to_exclude: int | None = None,
            exclude_last_period: bool = False,
            price_or_return: str = "price",
    ) -> pd.DataFrame:

        if price_or_return not in ("price", "return"):
            raise ValueError("price_or_return must be 'price' or 'return'")

        if exclude_last_period:
            if nb_period_to_exclude is None:
                raise ValueError("nb_period_to_exclude must be provided")
            end_shift = nb_period_to_exclude
        else:
            end_shift = 0

        start_shift = nb_period + end_shift

        if price_or_return == "price":
            mom = df.shift(end_shift) / df.shift(start_shift) - 1
        else:
            # cumulative return over the window from a returns series
            mom = (1 + df).rolling(nb_period).apply(lambda x: x.prod(), raw=True) - 1
            if end_shift > 0:
                mom = mom.shift(end_shift)

        return mom

