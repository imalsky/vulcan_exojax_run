"""Differentiable bridge: VULCAN-JAX (nz) pressure grid -> ExoJax ART (nlayer) grid.

The chemistry runs on VULCAN's vertical grid; ExoJax's ``ArtTransPure`` has its own
log-spaced pressure grid. We map per-layer profiles (T, VMR, mean molecular weight)
between them with a log-pressure linear interpolation that is pure JAX (``jnp.interp``)
and therefore differentiable -- so forward-mode tangents pass cleanly across the bridge.

``jnp.interp`` requires ascending sample points, so we sort the VULCAN grid by log10(P)
once (static index array) and clamp outside the range (the ART bounds are chosen inside
the VULCAN envelope in config, so clamping never actually triggers).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def make_to_art(p_bar_vulcan: np.ndarray, p_bar_art: np.ndarray):
    """Build a differentiable ``to_art(profile_nz) -> profile_nlayer`` interpolator.

    Parameters
    ----------
    p_bar_vulcan : (nz,) np.ndarray
        VULCAN cell-center pressures in bar (any monotonic order).
    p_bar_art : (nlayer,) np.ndarray
        ExoJax ART layer pressures in bar (top-to-bottom).

    Returns
    -------
    callable
        ``to_art(profile)`` where ``profile`` is a JAX array of shape (nz,);
        returns a JAX array of shape (nlayer,) interpolated in log-pressure.
    """
    order = np.argsort(p_bar_vulcan)                       # static
    logP_v_sorted = jnp.asarray(np.log10(p_bar_vulcan[order]))
    logP_art = jnp.asarray(np.log10(np.asarray(p_bar_art)))
    order_j = jnp.asarray(order)

    def to_art(profile_nz):
        fp = profile_nz[order_j]
        return jnp.interp(logP_art, logP_v_sorted, fp)

    return to_art
