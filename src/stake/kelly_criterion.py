def fractional_kelly(p_win: float, odds: float, bankroll: float, k: float = 0.5) -> float:
    """Return stake size using fractional Kelly."""
    b = odds - 1
    fraction = (p_win * b - (1 - p_win)) / b
    return max(0, fraction * k) * bankroll
