# -*- coding: utf-8 -*-
"""Generate structure figures for the simulation report:
   - structure_index.png   : real Lumerical index map of the meshed device
   - structure_zoom.png     : zoom on the slot + first grating teeth (index)
   - structure_schematic.png: annotated cross-section schematic
"""
import sys, math, os
LUMERICAL_ROOT = os.environ.get("LUMERICAL_ROOT", r"D:\Program Files\ANSYS Inc\v252")
sys.path.append(os.environ.get("LUMAPI_DIR", os.path.join(LUMERICAL_ROOT, "api", "python")))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrow
import lumapi, grating_coupler_fdtd as g

nm, um = g.nm, g.um
p = dict(g.PARAMS)
yl = g.y_levels(p)
run = p["h_etch"]/math.tan(math.radians(p["sidewall_deg"]))
RES = g.RESULTS_DIR

# ----------------------------------------------------------------------------
# 1) Real index map from Lumerical (add an index monitor, run engine once)
# ----------------------------------------------------------------------------
fdtd = lumapi.FDTD(hide=True)
layout = g.build_model(fdtd, p)
fdtd.addindex(); fdtd.set("name","index_mon"); fdtd.set("monitor type","2D Z-normal")
fdtd.set("x min", layout["wg_right"]-3*um); fdtd.set("x max", layout["grating_top_right"]+2*um)
fdtd.set("y min", yl["box_bot"]-0.3*um); fdtd.set("y max", yl["film_top"]+1.5*um)
fdtd.save(g.FSP_PATH)
g.run_sim(fdtd)
idx = fdtd.getresult("index_mon","index")
x = np.squeeze(np.array(idx["x"])); y = np.squeeze(np.array(idx["y"]))
n = np.real(np.squeeze(np.array(idx["index_z"])))     # Ez sees nz (=ne in LN)
fdtd.close()

def index_plot(xlim, fname, title, ylim=None, figsize=(9,4.2)):
    plt.figure(figsize=figsize)
    pc = plt.pcolormesh(x/um, y/um, n.T, shading="auto", cmap="viridis")
    plt.colorbar(pc, label="refractive index n (Ez / nz)")
    plt.xlabel("x  (propagation, um)"); plt.ylabel("y  (vertical, um)")
    plt.title(title); plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    plt.tight_layout(); plt.savefig(os.path.join(RES,fname), dpi=150); plt.close()

index_plot((x.min()/um, x.max()/um), "structure_index.png",
           "Index profile (n_z) of the 2D-FDTD model")
index_plot((layout["wg_right"]/um-1.5, layout["wg_right"]/um+3.5), "structure_zoom.png",
           "Zoom: waveguide / 187 nm slot / first grating teeth (70 deg sidewalls)",
           ylim=(-0.35, 1.05), figsize=(9,3.4))

# ----------------------------------------------------------------------------
# 2) Annotated schematic (drawn from parameters)
# ----------------------------------------------------------------------------
FFs = g.fill_factors(p)
fig, ax = plt.subplots(figsize=(11,5))
x0 = layout["wg_right"]/um - 3
x1 = layout["grating_top_right"]/um + 2
# layers
ax.add_patch(Rectangle((x0, (yl["si_bot"])/um), x1-x0, (yl["box_bot"]-yl["si_bot"])/um,
                       fc="#9aa0a6", ec="none")); ax.text(x1-1.5,(yl["box_bot"]-0.55*um)/um,"Si substrate",ha="right",fontsize=9)
ax.add_patch(Rectangle((x0, yl["box_bot"]/um), x1-x0, (yl["film_bot"]-yl["box_bot"])/um,
                       fc="#cfe8ff", ec="none")); ax.text(x1-1.5,(yl["box_bot"]/um+ (4.7)/2),"SiO2 BOX  (4.7 um)",ha="right",fontsize=9)
# LN slab
ax.add_patch(Rectangle((x0, yl["film_bot"]/um), x1-x0, (yl["trench_floor"]-yl["film_bot"])/um,
                       fc="#ffd27f", ec="k", lw=0.4))
# WG
ax.add_patch(Rectangle((x0, yl["film_bot"]/um), (layout["wg_right"]/um - x0),
                       (yl["film_top"]-yl["film_bot"])/um, fc="#f4b400", ec="k", lw=0.5))
# teeth
for i,FF in enumerate(FFs):
    wb,wt = g.tooth_widths(FF,p["period"],run,p["ff_reference"])
    xc = (i*p["period"] + FF*p["period"]/2.0)/um
    verts = [(xc-wb/2/um, yl["trench_floor"]/um),(xc+wb/2/um, yl["trench_floor"]/um),
             (xc+wt/2/um, yl["film_top"]/um),(xc-wt/2/um, yl["film_top"]/um)]
    ax.add_patch(Polygon(verts, closed=True, fc="#f4b400", ec="k", lw=0.5))
# fiber beam arrow (angle theta from +y, tilts toward -x as it goes down)
th = math.radians(p["fiber_angle_deg"])
xs = p["fiber_x"]/um; ys = (yl["film_top"]+p["fiber_gap"]+1.2*um)/um
L=2.2
ax.add_patch(FancyArrow(xs+L*math.sin(th), ys+L*math.cos(th), -L*math.sin(th), -L*math.cos(th),
             width=0.05, head_width=0.35, head_length=0.35, fc="#d93025", ec="#d93025", length_includes_head=True))
ax.text(xs+1.0, ys+0.3, "fiber Gaussian\n w0=5.2 um, theta=21deg", color="#d93025", fontsize=9)
# monitor line
xm = (layout["wg_right"]/um - 2.0)
ax.plot([xm,xm],[(yl["film_bot"]/um-0.8),(yl["film_top"]/um+0.8)],"b--",lw=1.5)
ax.text(xm-0.1, yl["film_top"]/um+1.0,"mode-expansion\nmonitor", color="b", ha="center", fontsize=8)
# annotations
ax.annotate("", xy=(0,(yl["film_top"]/um+0.45)), xytext=(p["period"]/um,(yl["film_top"]/um+0.45)),
            arrowprops=dict(arrowstyle="<->")); ax.text(p["period"]/2/um, yl["film_top"]/um+0.55, "Lambda=754 nm", ha="center", fontsize=8)
ax.text(x0+0.1, yl["film_top"]/um-0.25, "LN  600 nm", fontsize=9)
ax.text(0.3, (yl["trench_floor"]/um-0.28), "175 nm slab  (etch h=425 nm)", fontsize=8)
ax.set_xlim(x0,x1); ax.set_ylim(yl["si_bot"]/um-0.2, ys+2.0)
ax.set_xlabel("x  (propagation, um)"); ax.set_ylabel("y  (vertical, um)")
ax.set_title("Annotated cross-section (X-cut LNOI negative-angle apodized GC)")
plt.tight_layout(); plt.savefig(os.path.join(RES,"structure_schematic.png"), dpi=150); plt.close()
print("FIGS DONE")
