"""Provides conversion functionality for DP mechanisms."""

from math import log, sqrt


def epsilon_delta_dp_to_rho_zcdp(
    epsilon: float,
    delta: float
) -> float:
    """Convert (epsilon, delta)-DP values to rho-zCDP value

    Args:
        epsilon (float): epsilon privacy parameter.
        delta (float): delta privacy parameter.

    Returns:
        float: rho zCDP privacy parameter.
    """
    ldinv = log(1/delta)
    rho = epsilon + 2*(ldinv - sqrt(ldinv * (epsilon + ldinv)))
    return rho
