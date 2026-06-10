from __future__ import annotations

import numpy as np
import pandas as pd


def tail_metrics(returns: pd.Series) -> dict[str, float]:
    sample = returns.dropna()
    if sample.empty:
        return {"tail_loss_95": np.nan, "cvar_95": np.nan}
    q = sample.quantile(0.05)
    return {"tail_loss_95": float(q), "cvar_95": float(sample[sample <= q].mean())}
