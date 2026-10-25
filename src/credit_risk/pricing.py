"""Expected loss and an illustrative risk-based price.

Expected loss (EL) = PD x LGD x EAD
  PD  - lifetime probability of charge-off over the 36-month loan (our model's output)
  LGD - share of the outstanding balance lost when a loan charges off, after recoveries
  EAD - balance outstanding when it charges off

LGD and EAD are estimated from charged-off loans in the TRAINING years only, using fields that
are only known after the loan ends (principal repaid, recoveries). That is fine for estimating
loss assumptions, but those fields are never model inputs.

Because PD here is a lifetime figure, EL is a lifetime expected loss per loan. It is not an
IFRS 9 expected credit loss: there are no 12-month vs lifetime stages, no discounting and no
forward-looking economic scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_lgd_ead(train: pd.DataFrame) -> dict:
    """Average realised LGD and EAD (as a share of the original amount) on charged-off training loans."""
    co = train[(train["bad"] == 1) & train["funded_amnt"].gt(0)].copy()
    outstanding = (co["funded_amnt"] - co["total_rec_prncp"]).clip(lower=0)
    net_recovery = (co["recoveries"].fillna(0) - co["collection_recovery_fee"].fillna(0)).clip(lower=0)
    valid = outstanding > 0
    lgd = 1 - (net_recovery[valid] / outstanding[valid]).clip(upper=1)
    ead_share = outstanding / co["funded_amnt"]
    return {"lgd": float(lgd.mean()), "ead_share": float(ead_share.mean()), "charged_off_loans_used": int(valid.sum())}


def expected_loss(pd_, amount, lgd: float, ead_share: float) -> np.ndarray:
    return np.asarray(pd_) * lgd * ead_share * np.asarray(amount, dtype=float)


def el_sensitivity(pd_, amount, lgd_values=(0.7, 0.8, 0.9, 1.0), ead_values=(0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Total expected loss as a % of lending, for a range of LGD and EAD assumptions."""
    total = float(np.sum(amount))
    rows = [{"lgd": l, "ead_share": e, "expected_loss_pct_of_lending": float(expected_loss(pd_, amount, l, e).sum() / total)}
            for l in lgd_values for e in ead_values]
    return pd.DataFrame(rows)


def illustrative_rate(pd_, lgd: float, ead_share: float, term_years: float = 3.0, funding_cost=0.04,
                      operating_cost=0.02, target_margin=0.03) -> np.ndarray:
    """Annual rate that would cover expected loss, costs and a margin (a teaching example only).

    Expected lifetime loss is spread evenly over the loan's life. A real pricing model would also
    use amortisation, prepayment, competitor prices and capital costs.
    """
    annual_loss = np.asarray(pd_) * lgd * ead_share / term_years
    return funding_cost + operating_cost + target_margin + annual_loss
