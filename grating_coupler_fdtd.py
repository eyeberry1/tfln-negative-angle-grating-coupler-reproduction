# -*- coding: utf-8 -*-
"""
2D-FDTD reproduction (Lumerical 2025 R2 / lumapi) of:

  R. Liao et al., "High-efficiency thin-film lithium niobate apodized grating
  coupler utilizing negative diffraction angle," Opt. Express 32(26), 46299
  (2024). DOI 10.1364/OE.542031

Device: X-cut LNOI apodized grating coupler, NO bottom metal mirror.
Method: forward tilted-Gaussian "fiber" source + waveguide mode-expansion
monitor.  Coupling efficiency CE = power coupled into the fundamental guided
mode, normalized to source power.

Physics notes that MUST be respected for the absolute CE to come out right:
  * The upward directionality (~75 %, ~25 % to substrate per the paper Fig.3a)
    comes from the 4.7 um BOX acting as a Fabry-Perot cavity over the Si
    substrate -- there is no metal mirror.  Therefore the Si substrate is
    modeled and the bottom PML is placed INSIDE the Si, so the SiO2/Si Fresnel
    reflection and the BOX phase are captured.
  * X-cut LN, fundamental TE: the optical E-field lies along the device width
    (out of the 2D plane) = the LN optic axis (extraordinary index ne).  In
    Lumerical's 2D nomenclature an out-of-plane E-field is the "TM" label, so
    the Gaussian source injects Ez and the mode expansion uses "fundamental TM
    mode".

Compute path: lumapi's in-process fdtd.run() does NOT compute in this headless
environment (returns immediately, monitors empty), so the script saves the .fsp
and runs the standalone fdtd-engine.exe on it, then reloads. This runs on CPU:
Lumerical v252 GPU-FDTD supports 3D regions only ("GPU simulation only supports
3D simulation region"), so a 2D sim cannot use the GPU. A full 2D run is ~75 s.

Run (headless, no GUI) with the ANSYS python:
  & "D:\\Program Files\\ANSYS Inc\\v252\\python-3.13.1\\python.exe" grating_coupler_fdtd.py

Useful flags:
  --build-only   build geometry + save .fsp, do not run (smoke test)
  --quick        short simulation time + fewer freq points (fast sanity run)
  --tune         bounded local refinement of fiber angle/position around the
                 optimized defaults (writes results/tune_log.csv)
  --no-gpu       no-op (kept for CLI compatibility; 2D always runs on CPU)
"""

import os
import sys
import math
import argparse

import numpy as np

# --------------------------------------------------------------------------- #
#  lumapi import
# --------------------------------------------------------------------------- #
LUMERICAL_ROOT = os.environ.get("LUMERICAL_ROOT", r"D:\Program Files\ANSYS Inc\v252")
LUMAPI_DIR = os.environ.get("LUMAPI_DIR", os.path.join(LUMERICAL_ROOT, "api", "python"))
if LUMAPI_DIR not in sys.path:
    sys.path.append(LUMAPI_DIR)
import lumapi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
FSP_PATH = os.path.join(RESULTS_DIR, "grating_coupler.fsp")

nm = 1e-9
um = 1e-6

# --------------------------------------------------------------------------- #
#  Design parameters  (Table 1 of the paper + simulation choices)
# --------------------------------------------------------------------------- #
PARAMS = dict(
    # --- layer stack ---
    t_LN=600 * nm,          # LN film thickness
    h_etch=425 * nm,        # etch depth (leaves 600-425 = 175 nm slab)
    t_box=4.7 * um,         # buried oxide (BOX) thickness
    t_si=1.3 * um,          # modeled Si substrate thickness (bottom PML sits inside it)

    # --- grating ---
    n_periods=20,
    period=754 * nm,        # Lambda (fixed)
    FF1=0.58,               # first-period fill factor
    FF20=0.38,              # last-period fill factor
    sidewall_deg=70.0,      # sidewall inclination from horizontal (top-narrow/bottom-wide)
    ff_reference="mid",     # which height FF*Lambda refers to: 'mid' | 'top' | 'bottom'

    # --- anti-backscattering slot ---
    slot_top_width=187 * nm,
    slot_etch=425 * nm,     # default: same as grating etch (partial). set to t_LN for full etch.

    # --- input waveguide ---
    wg_extra=9.0 * um,      # how far the WG runs to -x beyond its right edge
                            # (wide enough that the tilted beam is not clipped by the -x PML)

    # --- fiber / Gaussian source ---
    # Lumerical's "angle theta" sign is opposite to the paper's incidence-angle
    # convention. A POSITIVE theta here realizes the paper's negative diffraction
    # angle: it phase-matches the m = -1 order to a guided mode propagating toward
    # the waveguide (-x). (theta < 0 couples the +1 order toward +x, away from the
    # waveguide -- verified by the field map.) theta is also the knob that sets the
    # peak wavelength (paper Fig. 5d realigns the peak by the fiber angle): theta =
    # 21 deg centres the peak at ~1550 nm here (the bare grating peaks ~1570 nm).
    fiber_angle_deg=21.0,
    mfd=10.4 * um,          # SMF-28 mode-field diameter -> w0 = mfd/2
    fiber_x=4.5 * um,       # beam center x over the (forward) grating  [TUNING KNOB]
    fiber_gap=1.2 * um,     # source plane height above the LN top surface

    # --- spectrum ---
    lam0=1.55 * um,
    lam_start=1.45 * um,
    lam_stop=1.65 * um,
    freq_points=101,

    # --- FDTD / mesh ---
    mesh_accuracy=2,
    dx_fine=15 * nm,        # uniform override mesh over the grating
    sim_time=5000e-15,      # max simulation time
    pml_layers=12,
    # NOTE: Lumerical v252 GPU-FDTD supports 3D regions ONLY ("GPU simulation
    # only supports 3D simulation region").  This device is simulated in 2D
    # (paper's method; the -2.1 dB / 80 nm acceptance is a 2D figure), so it
    # runs on CPU.  GPU cannot accelerate a 2D region in Lumerical.
    use_gpu=False,
)

# Convenience y-coordinates (origin y=0 at LN/BOX interface, +y = up)
def y_levels(p):
    y_film_bot = 0.0
    y_film_top = p["t_LN"]
    y_trench_floor = p["t_LN"] - p["h_etch"]      # 175 nm
    y_box_bot = -p["t_box"]
    y_si_bot = -p["t_box"] - p["t_si"]
    return dict(film_bot=y_film_bot, film_top=y_film_top,
                trench_floor=y_trench_floor, box_bot=y_box_bot, si_bot=y_si_bot)


# --------------------------------------------------------------------------- #
#  Materials
# --------------------------------------------------------------------------- #
def sellmeier_LN(lam_um):
    """Zelmon, Small & Jundt, JOSA B 14, 3319 (1997), congruent LiNbO3.
    Returns (no, ne) for wavelength(s) in micrometers."""
    l2 = np.asarray(lam_um, dtype=float) ** 2
    no2 = (1.0 + 2.6734 * l2 / (l2 - 0.01764)
           + 1.2290 * l2 / (l2 - 0.05914)
           + 12.614 * l2 / (l2 - 474.60))
    ne2 = (1.0 + 2.9804 * l2 / (l2 - 0.02047)
           + 0.5981 * l2 / (l2 - 0.06660)
           + 8.9543 * l2 / (l2 - 416.08))
    return np.sqrt(no2), np.sqrt(ne2)


def add_materials(fdtd, p):
    """Anisotropic (diagonal) dispersive LN: eps_xx=eps_yy=no^2, eps_zz=ne^2.
    The out-of-plane (z) axis is the LN optic axis, so the physical-TE field
    (Ez) sees ne."""
    c = 299792458.0
    lam = np.linspace(p["lam_start"], p["lam_stop"], 25)          # m
    lam_um = lam / um
    no, ne = sellmeier_LN(lam_um)
    f = c / lam                                                  # Hz
    # Sampled 3D data, diagonal anisotropy: columns [f, eps_xx, eps_yy, eps_zz]
    data = np.zeros((len(f), 4), dtype=complex)
    data[:, 0] = f
    data[:, 1] = no ** 2
    data[:, 2] = no ** 2
    data[:, 3] = ne ** 2

    name = "LN_xcut_aniso"
    try:
        fdtd.deletematerial(name)
    except Exception:
        pass
    tmp = fdtd.addmaterial("Sampled 3D data")      # returns the new material's name
    fdtd.setmaterial(tmp, "name", name)
    fdtd.setmaterial(name, "anisotropy", 1)        # 1 = diagonal
    fdtd.setmaterial(name, "sampled data", data)
    return name


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #
def tooth_widths(FF, period, run, reference):
    """Return (w_bottom, w_top) of a trapezoidal tooth for a given fill factor.
    `run` = h_etch / tan(sidewall) is the per-side bottom->top horizontal
    offset; top is narrower than bottom by 2*run."""
    w_ref = FF * period
    if reference == "mid":
        w_bot = w_ref + run
        w_top = w_ref - run
    elif reference == "top":
        w_top = w_ref
        w_bot = w_ref + 2 * run
    elif reference == "bottom":
        w_bot = w_ref
        w_top = w_ref - 2 * run
    else:
        raise ValueError("ff_reference must be 'mid', 'top' or 'bottom'")
    return w_bot, w_top


def fill_factors(p):
    n = p["n_periods"]
    dFF = p["FF1"] - p["FF20"]
    return [p["FF1"] - i * dFF / (n - 1) for i in range(n)]


def build_geometry(fdtd, p, mat_LN):
    yl = y_levels(p)
    run = p["h_etch"] / math.tan(math.radians(p["sidewall_deg"]))   # ~154.7 nm
    period = p["period"]
    FFs = fill_factors(p)

    # --- Si substrate ---
    fdtd.addrect()
    fdtd.set("name", "Si_substrate")
    fdtd.set("x min", -50 * um); fdtd.set("x max", 50 * um)
    fdtd.set("y min", yl["si_bot"]); fdtd.set("y max", yl["box_bot"])
    fdtd.set("material", "Si (Silicon) - Palik")

    # --- BOX ---
    fdtd.addrect()
    fdtd.set("name", "BOX")
    fdtd.set("x min", -50 * um); fdtd.set("x max", 50 * um)
    fdtd.set("y min", yl["box_bot"]); fdtd.set("y max", yl["film_bot"])
    fdtd.set("material", "SiO2 (Glass) - Palik")

    # --- continuous LN slab (175 nm) across the grating region ---
    fdtd.addrect()
    fdtd.set("name", "LN_slab")
    fdtd.set("x min", -50 * um); fdtd.set("x max", 50 * um)
    fdtd.set("y min", yl["film_bot"]); fdtd.set("y max", yl["trench_floor"])
    fdtd.set("material", mat_LN)

    # --- grating teeth (trapezoids on top of the slab) ---
    # period i occupies [x_g_start + (i-1)*period, ...]; tooth reference width
    # is left-aligned within the period, trapezoid centered on its reference span.
    x_g_start = 0.0
    tooth1_top_left = None
    for i, FF in enumerate(FFs):
        w_bot, w_top = tooth_widths(FF, period, run, p["ff_reference"])
        period_left = x_g_start + i * period
        xc = period_left + (FF * period) / 2.0          # tooth center
        yb = yl["trench_floor"]
        yt = yl["film_top"]
        verts = np.array([
            [xc - w_bot / 2.0, yb],
            [xc + w_bot / 2.0, yb],
            [xc + w_top / 2.0, yt],
            [xc - w_top / 2.0, yt],
        ])
        fdtd.addpoly()
        fdtd.set("name", "tooth_%d" % (i + 1))
        fdtd.set("x", 0.0); fdtd.set("y", 0.0)
        fdtd.set("vertices", verts)
        fdtd.set("material", mat_LN)
        if i == 0:
            tooth1_top_left = xc - w_top / 2.0

    grating_top_right = x_g_start + (p["n_periods"] - 1) * period \
        + (FFs[-1] * period) / 2.0 + tooth_widths(FFs[-1], period, run, p["ff_reference"])[1] / 2.0
    grating_bottom_right = grating_top_right + run + 1 * um   # generous right margin

    # --- input waveguide (full 600 nm LN on the -x side) ---
    # The slot (top width Ws) is the gap between the WG top-right edge and
    # tooth-1 top-left edge.
    wg_right = tooth1_top_left - p["slot_top_width"]
    fdtd.addrect()
    fdtd.set("name", "WG")
    fdtd.set("x min", wg_right - p["wg_extra"]); fdtd.set("x max", wg_right)
    fdtd.set("y min", yl["film_bot"]); fdtd.set("y max", yl["film_top"])
    fdtd.set("material", mat_LN)

    # If the slot is a FULL etch, carve the slab away inside the slot opening by
    # inserting an air (etch) rectangle. For a partial slot (default) the slab
    # already provides the 175 nm floor, so nothing extra is needed.
    if p["slot_etch"] >= p["t_LN"] - 1e-12:
        fdtd.addrect()
        fdtd.set("name", "slot_fulletch")
        fdtd.set("x min", wg_right); fdtd.set("x max", tooth1_top_left)
        fdtd.set("y min", yl["film_bot"]); fdtd.set("y max", yl["trench_floor"])
        fdtd.set("material", "etch")          # overrides slab -> air, mesh order below

    layout = dict(wg_right=wg_right, tooth1_top_left=tooth1_top_left,
                  grating_top_right=grating_top_right,
                  grating_bottom_right=grating_bottom_right, run=run)
    return layout


# --------------------------------------------------------------------------- #
#  FDTD region + mesh
# --------------------------------------------------------------------------- #
def add_fdtd(fdtd, p, layout):
    yl = y_levels(p)
    x_min = layout["wg_right"] - p["wg_extra"] + 0.5 * um
    x_max = layout["grating_bottom_right"] + 2.0 * um
    y_min = yl["si_bot"] + 0.2 * um            # bottom PML sits inside the Si
    y_max = yl["film_top"] + 2.5 * um          # air above for the tilted beam

    fdtd.addfdtd()
    fdtd.set("dimension", "2D")
    fdtd.set("x min", x_min); fdtd.set("x max", x_max)
    fdtd.set("y min", y_min); fdtd.set("y max", y_max)
    fdtd.set("background material", "etch")    # air (n=1)
    fdtd.set("mesh accuracy", p["mesh_accuracy"])
    fdtd.set("pml layers", p["pml_layers"])
    # Conformal mesh (CPU): resolves the 70-deg slanted sidewalls with sub-cell
    # accuracy (a staircase mesh over-fills the slanted walls and red-shifts the
    # spectrum). "conformal variant 1" applies conformal averaging to dielectrics.
    try:
        fdtd.set("mesh refinement", "conformal variant 1")
    except Exception:
        pass
    fdtd.set("simulation time", p["sim_time"])
    fdtd.set("auto shutoff min", p.get("autoshutoff", 1e-5))

    # Fine uniform override mesh over the LN grating (resolves slanted walls).
    fdtd.addmesh()
    fdtd.set("name", "grating_mesh")
    fdtd.set("x min", layout["wg_right"] - 0.5 * um)
    fdtd.set("x max", layout["grating_bottom_right"])
    fdtd.set("y min", yl["film_bot"] - 0.15 * um)
    fdtd.set("y max", yl["film_top"] + 0.15 * um)
    fdtd.set("override x mesh", 1); fdtd.set("dx", p["dx_fine"])
    fdtd.set("override y mesh", 1); fdtd.set("dy", p["dx_fine"])


# --------------------------------------------------------------------------- #
#  Source
# --------------------------------------------------------------------------- #
def add_source(fdtd, p):
    yl = y_levels(p)
    y_src = yl["film_top"] + p["fiber_gap"]
    w0 = p["mfd"] / 2.0

    fdtd.addgaussian()
    fdtd.set("name", "fiber")
    fdtd.set("injection axis", "y-axis")
    fdtd.set("direction", "Backward")          # propagate toward -y (into chip)
    fdtd.set("y", y_src)
    fdtd.set("x", p["fiber_x"])
    fdtd.set("x span", 26 * um)                # wide enough to hold the Gaussian
    # out-of-plane E (Ez) = physical TE for X-cut LN
    fdtd.set("polarization angle", 90.0)
    fdtd.set("angle theta", p["fiber_angle_deg"])
    # Gaussian beam definition
    fdtd.set("use scalar approximation", 1)
    try:
        fdtd.set("beam parameters", "Waist size and position")
    except Exception:
        pass
    fdtd.set("waist radius w0", w0)
    fdtd.set("distance from waist", 0.0)       # Rayleigh range ~55 um >> gap -> waist ~ at surface
    # source band: use the global source limits (don't let the source default to 400-700 nm)
    fdtd.setglobalsource("wavelength start", p["lam_start"])
    fdtd.setglobalsource("wavelength stop", p["lam_stop"])
    fdtd.select("fiber")
    fdtd.set("override global source settings", 0)


# --------------------------------------------------------------------------- #
#  Monitors
# --------------------------------------------------------------------------- #
def add_monitors(fdtd, p, layout):
    yl = y_levels(p)
    fdtd.setglobalmonitor("frequency points", p["freq_points"])
    fdtd.setglobalmonitor("use source limits", 1)

    # --- waveguide cross-section field monitor (X-normal line) ---
    x_mon = layout["wg_right"] - 2.0 * um
    fdtd.addpower()
    fdtd.set("name", "wg_mon")
    fdtd.set("monitor type", "2D X-normal")
    fdtd.set("x", x_mon)
    fdtd.set("y", yl["film_bot"] + p["t_LN"] / 2.0)
    fdtd.set("y span", 2.5 * um)

    # --- mode expansion on that monitor (fundamental TM = physical TE) ---
    fdtd.addmodeexpansion()
    fdtd.set("name", "wg_expansion")
    fdtd.set("monitor type", "2D X-normal")
    fdtd.set("x", x_mon)
    fdtd.set("y", yl["film_bot"] + p["t_LN"] / 2.0)
    fdtd.set("y span", 2.5 * um)
    fdtd.set("mode selection", "fundamental TM mode")
    fdtd.setexpansion("port", "wg_mon")
    fdtd.set("frequency points", p["freq_points"])
    try:
        fdtd.set("use source limits", 1)
    except Exception:
        pass

    # --- substrate (downward) power monitor: directionality diagnostic ---
    fdtd.addpower()
    fdtd.set("name", "sub_mon")
    fdtd.set("monitor type", "2D Y-normal")
    fdtd.set("y", yl["box_bot"] + 0.2 * um)     # just above BOX/Si interface
    fdtd.set("x", 0.5 * (layout["wg_right"] + layout["grating_top_right"]))
    fdtd.set("x span", layout["grating_top_right"] - layout["wg_right"] + 8 * um)

    # --- field profile monitor (for the |E| image in the report) ---
    fdtd.addpower()
    fdtd.set("name", "field_xy")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x min", layout["wg_right"] - 3 * um)
    fdtd.set("x max", layout["grating_top_right"] + 3 * um)
    fdtd.set("y min", yl["box_bot"]); fdtd.set("y max", yl["film_top"] + 2 * um)


# --------------------------------------------------------------------------- #
#  Assemble
# --------------------------------------------------------------------------- #
def build_model(fdtd, p):
    fdtd.switchtolayout()
    fdtd.deleteall()
    mat_LN = add_materials(fdtd, p)
    layout = build_geometry(fdtd, p, mat_LN)
    add_fdtd(fdtd, p, layout)
    add_source(fdtd, p)
    add_monitors(fdtd, p, layout)
    if p["use_gpu"]:
        try:
            fdtd.setresource("FDTD", 1, "device type", "GPU")
            print("[info] GPU resource enabled.")
        except Exception as e:
            print("[warn] could not enable GPU (%s); using CPU." % e)
    fdtd.save(FSP_PATH)
    return layout


# --------------------------------------------------------------------------- #
#  Extraction
# --------------------------------------------------------------------------- #
def _expansion_TbTf(fdtd, name="wg_expansion"):
    """Return (lam_m, T_forward, T_backward) arrays from the mode expansion."""
    r = fdtd.getresult(name, "expansion for port")
    lam = np.squeeze(np.array(r["lambda"]))
    Tf = np.abs(np.squeeze(np.array(r["T_forward"])))
    Tb = np.abs(np.squeeze(np.array(r["T_backward"])))
    return lam, Tf, Tb


def extract_ce(fdtd):
    lam, Tf, Tb = _expansion_TbTf(fdtd)
    # The guided wave travels into the WG (toward -x). Whichever of T_forward /
    # T_backward carries the coupled power is the CE; report both, pick the max.
    if Tf is None and Tb is None:
        raise RuntimeError("mode expansion returned no T_forward/T_backward")
    cand = [a for a in (Tf, Tb) if a is not None]
    ce = np.maximum.reduce(cand) if len(cand) > 1 else cand[0]
    lam = np.asarray(lam, float).ravel()
    ce = np.asarray(ce, float).ravel()
    order = np.argsort(lam)
    return lam[order], ce[order], (Tf, Tb)


def metrics(lam, ce):
    ce = np.clip(ce, 1e-12, None)
    ce_db = 10 * np.log10(ce)
    ip = int(np.argmax(ce_db))
    peak_db = ce_db[ip]
    lam_peak = lam[ip]
    # 3-dB bandwidth around the peak
    thr = peak_db - 3.0
    above = ce_db >= thr
    bw = float("nan")
    if above.any():
        idx = np.where(above)[0]
        lo, hi = idx[0], idx[-1]
        def cross(i0, i1):
            if i0 == i1:
                return lam[i0]
            return np.interp(thr, [ce_db[i0], ce_db[i1]], [lam[i0], lam[i1]])
        l_lo = cross(max(lo - 1, 0), lo)
        l_hi = cross(hi, min(hi + 1, len(lam) - 1))
        bw = abs(l_hi - l_lo)
    return dict(peak_db=peak_db, lam_peak=lam_peak, bw=bw, ce_db=ce_db, ip=ip)


def substrate_loss(fdtd, lam, ip):
    """Fraction of source power radiated downward into the substrate, evaluated
    AT the peak wavelength (index ip). The sub_mon is a Y-normal monitor below
    the BOX; its transmission is negative (downward) so we take the magnitude."""
    try:
        T = np.abs(np.squeeze(np.array(fdtd.transmission("sub_mon"))))
        # align to the (sorted) CE wavelength axis
        lam_sub = np.squeeze(np.array(fdtd.getresult("sub_mon", "T")["lambda"]))
        order = np.argsort(lam_sub)
        T = T[order]
        return float(T[ip])
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
#  Run helpers
# --------------------------------------------------------------------------- #
ENGINE = os.environ.get(
    "LUMERICAL_ENGINE", os.path.join(LUMERICAL_ROOT, "bin", "fdtd-engine.exe")
)


def run_sim(fdtd):
    """Run via the standalone engine. In this headless environment the in-process
    fdtd.run() returns without populating monitors, so we save -> run the engine
    on the .fsp -> reload. The GPU/CPU resource configured on the FDTD object is
    written into the .fsp and honored by the engine."""
    import subprocess
    fdtd.switchtolayout()
    fdtd.save(FSP_PATH)
    proc = subprocess.run([ENGINE, FSP_PATH], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout[-2000:])
        raise RuntimeError("fdtd-engine exited with code %d" % proc.returncode)
    fdtd.load(FSP_PATH)


def evaluate(fdtd, p, build=True):
    if build:
        build_model(fdtd, p)
    run_sim(fdtd)
    lam, ce, (Tf, Tb) = extract_ce(fdtd)
    m = metrics(lam, ce)
    m["sub_loss"] = substrate_loss(fdtd, lam, m["ip"])
    m["lam"] = lam
    m["ce"] = ce
    return m


# --------------------------------------------------------------------------- #
#  Output
# --------------------------------------------------------------------------- #
def save_spectrum_csv(lam, ce, path):
    ce = np.clip(ce, 1e-12, None)
    ce_db = 10 * np.log10(ce)
    with open(path, "w") as f:
        f.write("wavelength_nm,CE,CE_dB\n")
        for l, c, d in zip(lam, ce, ce_db):
            f.write("%.4f,%.6f,%.4f\n" % (l / nm, c, d))


def make_plots(fdtd, lam, ce):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[warn] matplotlib unavailable (%s); skipping plots." % e)
        return
    ce_db = 10 * np.log10(np.clip(ce, 1e-12, None))
    plt.figure(figsize=(6, 4))
    plt.plot(lam / nm, ce_db, "-b")
    plt.axhline(-2.1, color="r", ls="--", lw=0.8, label="target -2.1 dB")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Coupling efficiency (dB)")
    plt.title("LNOI negative-angle apodized GC"); plt.grid(True, alpha=0.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ce_spectrum.png"), dpi=160)
    plt.close()

    try:
        E = fdtd.getresult("field_xy", "E")
        x = np.squeeze(np.array(E["x"])); y = np.squeeze(np.array(E["y"]))
        Efield = np.array(E["E"])               # [x,y,z,f,component]
        Emag = np.sqrt(np.sum(np.abs(Efield) ** 2, axis=-1))
        Emag = np.squeeze(Emag)
        if Emag.ndim == 3:
            Emag = Emag[:, :, Emag.shape[2] // 2]
        plt.figure(figsize=(7, 4))
        plt.pcolormesh(x / um, y / um, Emag.T, shading="auto", cmap="inferno")
        plt.xlabel("x (um)"); plt.ylabel("y (um)"); plt.title("|E| field")
        plt.colorbar(label="|E|"); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "efield.png"), dpi=160)
        plt.close()
    except Exception as e:
        print("[warn] field plot skipped (%s)" % e)


def report(m, p):
    print("\n" + "=" * 64)
    print(" LNOI negative-angle apodized grating coupler  --  2D-FDTD result")
    print("=" * 64)
    print(" ff_reference   : %s" % p["ff_reference"])
    print(" fiber x / angle: %.3f um / %.2f deg" % (p["fiber_x"] / um, p["fiber_angle_deg"]))
    print(" peak CE        : %.2f dB  (%.1f %%)" %
          (m["peak_db"], 100 * 10 ** (m["peak_db"] / 10)))
    print(" peak wavelength: %.1f nm" % (m["lam_peak"] / nm))
    print(" 3-dB bandwidth : %.1f nm" % (m["bw"] / nm))
    print(" substrate loss : %.1f %%" % (100 * m["sub_loss"]))
    target_db = -2.1
    ok = (abs(m["peak_db"] - target_db) <= 0.7) and (1500 < m["lam_peak"] / nm < 1600)
    print(" acceptance     : %s (target -2.1 dB near 1550-1552 nm, BW ~80 nm)"
          % ("PASS" if ok else "not yet"))
    print("=" * 64 + "\n")
    return ok


# --------------------------------------------------------------------------- #
#  Auto-tune (bounded)
# --------------------------------------------------------------------------- #
def auto_tune(fdtd, p):
    log_path = os.path.join(RESULTS_DIR, "tune_log.csv")
    log = open(log_path, "w")
    log.write("ff_ref,fiber_x_um,angle_deg,peak_dB,lam_peak_nm,bw_nm,sub_loss\n")

    best = None       # (peak_dB, params, metrics)

    def trial(pp, tag):
        nonlocal best
        m = evaluate(fdtd, pp, build=True)
        log.write("%s,%.3f,%.3f,%.4f,%.2f,%.2f,%.4f\n" % (
            pp["ff_reference"], pp["fiber_x"] / um, pp["fiber_angle_deg"],
            m["peak_db"], m["lam_peak"] / nm, m["bw"] / nm, m["sub_loss"]))
        log.flush()
        print("[tune] %-16s peak=%.3f dB  lam=%.1f nm  BW=%.1f nm  sub=%.0f%%" % (
            tag, m["peak_db"], m["lam_peak"] / nm, m["bw"] / nm, 100 * m["sub_loss"]))
        # prefer the highest CE whose peak sits in the 1548-1554 nm window
        in_band = 1548 <= m["lam_peak"] / nm <= 1554
        score = m["peak_db"] - (0.0 if in_band else 1.0)
        if best is None or score > best[0]:
            best = (score, dict(pp), m)
        return m

    # Local refinement around the optimized defaults. theta sets the peak
    # wavelength (~0.7 nm per 0.1 deg here) and fiber_x trims the mode overlap.
    for dth in (-0.4, -0.2, 0.0, 0.2, 0.4):
        for dx in (-0.5 * um, 0.0, 0.5 * um):
            pp = dict(p)
            pp["fiber_angle_deg"] = p["fiber_angle_deg"] + dth
            pp["fiber_x"] = p["fiber_x"] + dx
            trial(pp, "th%+.1f_x%+.1f" % (dth, dx / um))

    log.close()
    print("[tune] best peak = %.3f dB @ %.1f nm  (theta=%.1f deg, x=%.2f um)" % (
        best[2]["peak_db"], best[2]["lam_peak"] / nm,
        best[1]["fiber_angle_deg"], best[1]["fiber_x"] / um))
    return best[1], best[2]


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    p = dict(PARAMS)
    if args.no_gpu:
        p["use_gpu"] = False
    if args.quick:
        p["sim_time"] = 800e-15
        p["freq_points"] = 31

    fdtd = lumapi.FDTD(hide=True)
    try:
        if args.build_only:
            layout = build_model(fdtd, p)
            print("[ok] model built and saved to %s" % FSP_PATH)
            print("[ok] layout: %s" % {k: (round(v / nm, 1) if isinstance(v, float) else v)
                                       for k, v in layout.items()})
            return

        if args.tune:
            p_best, m = auto_tune(fdtd, p)
            p.update(p_best)
        else:
            m = evaluate(fdtd, p, build=True)

        save_spectrum_csv(m["lam"], m["ce"], os.path.join(RESULTS_DIR, "ce_spectrum.csv"))
        make_plots(fdtd, m["lam"], m["ce"])
        report(m, p)
        fdtd.save(FSP_PATH)
    finally:
        fdtd.close()


if __name__ == "__main__":
    main()
