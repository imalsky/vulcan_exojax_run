"""VULCAN-JAX side of the demo: the differentiable physics-parameters -> converged-VMR map.

``build_chem_model(profile)`` runs the one-time WASP-39b pre-loop + a single warm-up
convergence (which compiles & caches the JIT'd inner runner), then returns a model whose
``converged_ymix(theta)`` re-converges the closed column as a function of

    theta = [lnZ, c_o, lnKzz, T_int]   (all scalars)

Each knob reuses a *validated* jax_paper pattern (all forward-mode, all FD-checked there):
  * lnZ    -> scale metal-bearing initial abundances   (fig_metallicity_sens.py)
  * c_o    -> scale carbon-bearing initial abundances   (carbon-enrichment proxy for C/O)
  * lnKzz  -> scale the eddy-diffusion profile           (fig_kzz_jvp.py)
  * T_int  -> uniform T shift with rates rebuilt on-graph (fig_so2_temperature.py)

Closed-column note: the runner forgets the initial speciation except through the conserved
elemental column totals, so metallicity / carbon enrichment are correctly expressed as
initial-abundance (y0) directions. FastChem (which set the EQ y0) stays frozen/off-graph;
we perturb y0 directly, exactly as the validated metallicity script does.

The runner's lax.while_loop supports jvp/jacfwd but NOT vjp, so forward-mode is the
end-to-end route -- which is also optimal here (4 scalar inputs -> high-dim spectrum).
"""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import numpy as np

import config

# --- env + path setup MUST happen before importing vulcan_jax ---------------
os.environ["VULCAN_JAX_NETWORK"] = config.VULCAN_NETWORK
os.environ["VULCAN_JAX_ATOM_LIST"] = config.VULCAN_ATOM_LIST
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.chdir(str(config.JAXROOT))
sys.path.insert(0, str(config.JAXROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def build_chem_model(profile: dict, tp_eval=None, n_tp_params: int = 0) -> SimpleNamespace:
    """Build the converged WASP-39b model and the differentiable converged_ymix(theta).

    Parameters
    ----------
    profile : dict
        One of ``config.SMOKE`` / ``config.FULL`` -- supplies ``use_photo`` and
        ``yconv_cri``.
    tp_eval : callable or None, optional
        Temperature-profile hook. When ``None`` (default, unchanged behavior) the
        temperature is the validated uniform shift ``T = T_base + theta[3]`` (theta[3]
        is ``T_int``). When supplied, ``tp_eval(theta[3:3+n_tp_params], p_bar)`` returns
        the full (nz,) T-P profile (bar-indexed) that replaces the scalar shift -- used
        by the retrieval framework to retrieve an ExoJax Guillot/power-law T-P. The
        rest of the chemistry (rate rebuild on-graph, n_0, Ti, pv carry) is identical.
    n_tp_params : int, optional
        Number of T-P parameters consumed from ``theta[3:]`` when ``tp_eval`` is given.

    Returns
    -------
    SimpleNamespace with fields:
        converged_ymix(theta) -> (nz, ni) linear VMR, float64, differentiable
        T_base   : (nz,) baseline temperature (np.float64)
        p_bar    : (nz,) pressure grid in bar (np.float64)
        sidx     : dict species-name -> column index
        species_masses : (ni,) jnp molar mass per species (g/mol)
        nz, ni   : ints
    """
    t0 = time.time()
    import importlib

    # Baseline VULCAN config module: overridable per profile (the retrieval framework's
    # case presets set this) with the shared W39b default for all legacy consumers.
    cfg = importlib.import_module(profile.get("vulcan_cfg_module") or config.W39B_CFG_MODULE)
    cfg.use_live_plot = cfg.use_live_flux = cfg.use_print_prog = False
    cfg.use_photo = bool(profile["use_photo"])
    cfg.yconv_cri = float(profile["yconv_cri"])
    if profile.get("nz"):
        cfg.nz = int(profile["nz"])
    if profile.get("count_min"):
        cfg.count_min = int(profile["count_min"])
    if profile.get("count_max"):
        cfg.count_max = int(profile["count_max"])
    # Warm-continuation step cap (accepted steps) for the MUTATION path only. A warm
    # re-converge from a particle's own converged column normally needs ~count_min-300
    # steps; a proposal still not converged at warm_count_max is headed for rejection
    # anyway, so cutting the loop there (instead of at the full cold count_max) stops a
    # single bad lane from dragging the whole lockstep batch through thousands of wasted
    # steps. Realized via a SECOND runner whose _Statics snapshot the smaller cap (the
    # cap is baked into the jitted while_loop at trace time); the cold/two-stage solves
    # keep count_max. warm_count_max == count_max (or absent) means one shared runner.
    warm_count_max = int(profile.get("warm_count_max") or cfg.count_max)
    if warm_count_max > int(cfg.count_max):
        raise ValueError(
            f"warm_count_max={warm_count_max} exceeds count_max={int(cfg.count_max)}: "
            "the warm mutation cap must be at most the cold cap (it exists to REJECT "
            "doomed proposals earlier, not to extend them)")
    if profile.get("dt_max"):               # physical step-size cap (prevents the dt-balloon
        cfg.dt_max = float(profile["dt_max"])  # non-convergence at high Kzz; see config_schema)
    if profile.get("yconv_min"):           # close the loose convergence OR-branch (default 0.1)
        cfg.yconv_min = float(profile["yconv_min"])
    if profile.get("slope_cri"):
        cfg.slope_cri = float(profile["slope_cri"])
    if profile.get("fastchem_met_scale"):  # BASELINE metallicity (x solar); W39b default 10.0.
        cfg.fastchem_met_scale = float(profile["fastchem_met_scale"])  # build at the bottom -> march up
    # Generic cfg overrides (e.g. use_moldiff=False for the no-transport equilibrium tier).
    # Applied BEFORE the pre-loop build, so they reach make_atm_static / OuterLoop exactly
    # like use_photo does. The fisher_zco tier configs are the only users.
    for _k, _v in (profile.get("cfg_overrides") or {}).items():
        setattr(cfg, _k, _v)

    from vulcan_jax.state import RunState, legacy_view
    from vulcan_jax import network as net_mod, composition, rates_jax
    from vulcan_jax.jax_step import make_atm_static
    from vulcan_jax.gibbs import load_nasa9
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.phy_const import kb
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop

    rs = RunState.with_pre_loop_setup(cfg)
    var, atm, para = legacy_view(rs)
    network = net_mod.parse_network(str(resolve_data_path(cfg.network)))
    nz, ni = atm.Tco.shape[0], network.ni
    sidx = dict(network.species_idx)

    pco = jnp.asarray(np.asarray(atm.pco, dtype=np.float64))
    p_bar = np.asarray(atm.pco, dtype=np.float64) / 1.0e6
    p_bar_j = jnp.asarray(p_bar)   # bar-indexed grid for the optional tp_eval hook

    thermo_dir = resolve_data_path(cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = config.JAXROOT / "src" / "vulcan_jax" / "thermo"
    nasa9, _ = load_nasa9(network.species, thermo_dir)
    remove_list = getattr(cfg, "remove_list", None)

    # --- one warm-up run: compiles/caches integ._runner and confirms the primal converges
    solver = op_jax.Ros2JAX()
    if rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
    solver.naming_solver(para)
    print(f"[chem] setup {time.time() - t0:.1f}s; nz={nz} ni={ni} photo={cfg.use_photo}; "
          f"warming up runner ...", flush=True)
    tw = time.time()
    _ = integ(rs)
    print(f"[chem] warm-up converge {time.time() - tw:.1f}s", flush=True)

    # Warm-capped twin runner for the mutation path. OuterLoop._Statics snapshots
    # int(cfg.count_max) at _ensure_runner time, so the temporary mutation is safe: the
    # smaller cap is frozen into integ_warm's while_loop and cfg is restored right after.
    # Host-side closure construction only -- no extra XLA compile (the retrieval traces
    # integ_warm._runner inside its own jitted evaluators, exactly like integ._runner).
    if warm_count_max != int(cfg.count_max):
        _cold_cap = int(cfg.count_max)
        cfg.count_max = warm_count_max
        integ_warm = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
        integ_warm._ensure_runner(var, atm)
        cfg.count_max = _cold_cap
    else:
        integ_warm = integ

    atm_static = make_atm_static(atm, ni, nz, cfg=integ._cfg)
    state0 = integ._pack_state_from_runstate(rs)
    y0 = state0.y
    Kzz0 = atm_static.Kzz
    pv0 = state0.pv
    T_base = jnp.asarray(np.asarray(atm.Tco, dtype=np.float64))

    # --- composition masks for the y0 knobs -------------------------------
    compo = np.asarray(composition.compo_array)
    metal_cols = [config.ATOM_COLS[a] for a in ("O", "C", "N", "S")]
    metal_mask = jnp.asarray((compo[:, metal_cols].sum(axis=1) > 0).astype(np.float64))  # scale C/N/O/S; H,He fixed
    carbon_mask = jnp.asarray((compo[:, config.ATOM_COLS["C"]] > 0).astype(np.float64))  # C/O proxy
    # fixed-O C/O mode ("co_mode": "fixed_O"): every C atom lives in a C-bearing species,
    # and the O-carriers holding no C (H2O, OH, O2, SO, SO2, NO, ...) are disjoint from
    # them -- the two masks partition all O between "dragged along by C-carriers" and
    # "free to compensate".
    nO_per_species = jnp.asarray(np.asarray(compo[:, config.ATOM_COLS["O"]], dtype=np.float64))
    o_only_mask = jnp.asarray(((compo[:, config.ATOM_COLS["O"]] > 0)
                               & (compo[:, config.ATOM_COLS["C"]] == 0)).astype(np.float64))
    co_fixed_o = str(profile.get("co_mode", "proxy")) == "fixed_O"
    atomic_masses = jnp.asarray(np.asarray(config.ATOMIC_MASSES, dtype=np.float64))
    species_masses = jnp.asarray(np.asarray(compo, dtype=np.float64)) @ atomic_masses  # (ni,)
    # runner's own (ni, n_atoms) composition table, columns in its internal _atom_order --
    # used to rebuild the conserved atom totals (atom_ini) in the runner's exact basis.
    compo_run = jnp.asarray(np.asarray(integ._compo_arr, dtype=np.float64))
    # Opt-in: re-anchor the conserved atom totals to the perturbed column (needed for finite
    # metallicity/C-O steps; see _prep). OFF by default so the Fisher/sensitivity demo
    # (config.FULL) keeps its exact published behavior; the Hessian campaign turns it on.
    reanchor_atom_ini = bool(profile.get("reanchor_atom_ini", False))
    # Opt-in: zero the eddy-diffusion profile entirely (the no-transport equilibrium tier;
    # combine with cfg_overrides={"use_moldiff": False} so Dzz is off too). lnKzz is then inert.
    zero_kzz = bool(profile.get("zero_Kzz", False))

    co_bz_bound = float("inf")   # proxy mode has no b_z compensation -> no bound
    if co_fixed_o:
        # Build-time diagnostics for the fixed-O C/O knob: baseline C/O, how much of the
        # column's O sits in C-carriers (sets the b_z compensation), and the worst-layer
        # O-only share (b_z blows up where O-only carriers vanish).
        _y0n = np.asarray(y0, dtype=np.float64)
        _nC = np.asarray(compo[:, config.ATOM_COLS["C"]], dtype=np.float64)
        _nO = np.asarray(compo[:, config.ATOM_COLS["O"]], dtype=np.float64)
        _mC = np.asarray(carbon_mask); _mOo = np.asarray(o_only_mask)
        _C_tot = float((_y0n * _nC[None, :]).sum())
        _O_tot = float((_y0n * _nO[None, :]).sum())
        _OC_z = (_y0n * (_nO * _mC)[None, :]).sum(axis=1)
        _OO_z = (_y0n * (_nO * _mOo)[None, :]).sum(axis=1)
        with np.errstate(divide="ignore"):
            co_bz_bound = float(np.log(1.0 + np.min(_OO_z / _OC_z)))
        print(f"[chem] fixed-O C/O knob: baseline C/O = {_C_tot/_O_tot:.4f} "
              f"(ln = {np.log(_C_tot/_O_tot):+.4f}); O-in-C-carriers share "
              f"median {np.median(_OC_z/(_OC_z+_OO_z)):.3f}, max {np.max(_OC_z/(_OC_z+_OO_z)):.3f} "
              f"(b_z stays positive for c_o < {co_bz_bound:.2f})", flush=True)

    def _prep(theta, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0):
        """Build the perturbed initial runner state + atm for theta=[lnZ, c_o, lnKzz, T_int].

        Continuation: pass warm_y = a previously-CONVERGED y (and its lnZ_ref / c_o_ref) to
        warm-start from there instead of the fixed baseline y0. The metallicity AND C/O scales
        are then applied INCREMENTALLY (lnZ - lnZ_ref, c_o - c_o_ref), so a large absolute
        perturbation is reached by small steps from a nearby converged state -- which avoids
        the runner's snap-back-to-baseline, and (for C/O) avoids double-applying the C/O
        scaling to a warm_y that already carries it (a fixed-C/O metallicity march passes
        c_o_ref = c_o so the C/O factor is a no-op after the first step)."""
        lnZ, c_o, lnKzz = theta[0], theta[1], theta[2]
        c_o_inc = c_o - c_o_ref     # incremental C/O relative to the warm state

        # Temperature: default is the validated uniform T shift theta[3] (T_int); with a
        # tp_eval hook the full differentiable T-P profile theta[3:3+n_tp_params] is used
        # instead. Either way rate constants are rebuilt ON-GRAPH (rates_jax), with
        # n_0 = pco/(kb T), Ti, and the pv carry (fig_so2_temperature pattern).
        if tp_eval is None:
            T = T_base + theta[3]
        else:
            T = tp_eval(theta[3:3 + n_tp_params], p_bar_j)
        M = pco / (kb * T)
        k_arr = rates_jax.build_rate_array(network, T, M, nasa9, remove_list)
        Ti = 0.5 * (T[:-1] + T[1:])
        Kzz_eff = Kzz0 * 0.0 if zero_kzz else Kzz0 * jnp.exp(lnKzz)
        atm_T = atm_static._replace(Tco=T, Ti=Ti, M=M, Kzz=Kzz_eff)

        # Z + C/O: multiplicative y0 directions on the conserved element totals. From the
        # baseline (warm_y=None) the full lnZ is applied; in continuation only (lnZ - lnZ_ref).
        base = y0 if warm_y is None else warm_y
        if co_fixed_o:
            # c_o == delta ln(C/O) at fixed O, EXACTLY, layer by layer: scaling every
            # C-bearing species by e^c multiplies each layer's C total by e^c (all C lives
            # there); the O those species drag along (CO, CO2, ...) is compensated by
            # scaling the O-only carriers by b_z = 1 + (1 - e^c)*O_Ccarriers/O_Oonly, which
            # keeps each layer's O total invariant. Leakage is only into H (via H2O's H,
            # ~1e-3 relative per unit c) and N/S (via trace NO/SO/SO2 in the equilibrium
            # init). Smooth in c_o -> AD-safe; b_z > 0 within the range printed at build.
            OC_z = (base * (nO_per_species * carbon_mask)[None, :]).sum(axis=1)
            OO_z = (base * (nO_per_species * o_only_mask)[None, :]).sum(axis=1)
            b_z = 1.0 + (1.0 - jnp.exp(c_o_inc)) * OC_z / OO_z                # (nz,)
            cofac = jnp.where(carbon_mask[None, :] > 0, jnp.exp(c_o_inc), 1.0)  # (1, ni)
            cofac = jnp.where(o_only_mask[None, :] > 0, b_z[:, None], cofac)  # (nz, ni)
            y0p = base * jnp.exp((lnZ - lnZ_ref) * metal_mask)[None, :] * cofac
        else:
            scale = jnp.exp((lnZ - lnZ_ref) * metal_mask + c_o_inc * carbon_mask)  # (ni,)
            y0p = base * scale[None, :]
        ymix0 = y0p / jnp.sum(y0p, axis=1, keepdims=True)

        # CRITICAL (opt-in): re-anchor the conserved atom totals to the PERTURBED column. The
        # runner's atom-conservation (_compute_atom_loss) measures drift from pv.atom_ini; if
        # atom_ini stays at the baked baseline, finite metallicity/C-O steps that exceed the
        # loss threshold get the added metals driven back to baseline (the "snap to baseline"
        # seen for lnZ >= +0.10). Rebuilding atom_ini = sum_z compo^T y0p in the runner's own
        # basis makes the column conserve its own higher totals. Smooth in theta -> AD-safe.
        if reanchor_atom_ini:
            atom_ini_new = jnp.einsum("zi,ia->a", y0p, compo_run)  # (n_atoms,) runner order
            pv_T = pv0._replace(n_0=M, r_Tco=T, atom_ini=atom_ini_new)
        else:
            pv_T = pv0._replace(n_0=M, r_Tco=T)

        init = state0._replace(y=y0p, ymix=ymix0, k_arr=k_arr, pv=pv_T)
        return init, atm_T

    def converged_ymix(theta):
        """Re-converge the WASP-39b column under theta=[lnZ, c_o, lnKzz, T_int].

        Returns linear VMR (nz, ni). Differentiable end-to-end via forward-mode.
        """
        init, atm_T = _prep(theta)
        final = integ._runner(init, atm_T)
        return final.y / jnp.sum(final.y, axis=1, keepdims=True)

    def run_diag(theta):
        """Diagnostic twin of converged_ymix: returns (final_runner_state, init_state).

        Lets a caller inspect convergence (longdy/accept_count/t), whether the runner
        actually moved off the init, and whether the metallicity perturbation changed the
        conserved element totals. Not on any AD path."""
        init, atm_T = _prep(theta)
        final = integ._runner(init, atm_T)
        return final, init

    def converged_y(theta, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0, return_diag=False,
                    warm_cap=False):
        """Converged ABSOLUTE number densities y (nz, ni), with optional continuation
        warm-start (warm_y at lnZ_ref / c_o_ref). Differentiable via forward-mode w.r.t. theta.
        The SO2 column number density is then jnp.sum(y[:, so2] * dz); jvp gives both y (for
        chaining the next continuation step) and its lnZ-derivative (the index) in one pass.

        ``return_diag=True`` additionally returns ``final.accept_count`` (int32 scalar) so a
        caller can detect a count_max-exhausted (not-actually-converged) solve without
        re-deriving the runner's own termination ladder. AD-safe inside a forward-mode jvp
        chain: accept_count rides the runner's primal carry (no extra work) and, being
        integer-valued, carries no tangent -- callers on an AD path should wrap it in
        ``stop_gradient`` and cast, and must not differentiate w.r.t. it.

        ``warm_cap=True`` runs the warm-capped twin runner (count_max=warm_count_max) --
        the SMC mutation path, where a proposal that hasn't converged in warm_count_max
        steps is rejected rather than marched to the full cold cap."""
        init, atm_T = _prep(jnp.asarray(theta, dtype=jnp.float64), warm_y=warm_y,
                            lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)
        final = (integ_warm if warm_cap else integ)._runner(init, atm_T)
        if return_diag:
            return final.y, final.accept_count
        return final.y

    return SimpleNamespace(
        converged_ymix=converged_ymix,
        run_diag=run_diag,
        converged_y=converged_y,
        co_bz_bound=co_bz_bound,   # fixed-O knob validity: b_z > 0 iff c_o < this (baseline column)
        y0=np.asarray(y0, dtype=np.float64),   # baked baseline column (warm-start fallback)
        compo_array=compo,
        T_base=np.asarray(T_base),
        p_bar=p_bar,
        dz=np.asarray(atm.dz, dtype=np.float64),   # layer thickness (cm); for n0*dz column weighting
        sidx=sidx,
        species_masses=species_masses,
        nz=nz, ni=ni,
        count_max=int(cfg.count_max),   # the resolved (profile-overridden or module-default) cap
        warm_count_max=warm_count_max,  # mutation-path cap (== count_max when no twin runner)
    )
