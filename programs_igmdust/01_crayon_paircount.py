#!/usr/bin/env python3
"""
01_crayon_paircount.py — igmdust/DES: standard-crayon reddening estimator.

DES Y3 redMaGiC highlum_highz galaxies as "standard crayons": red-sequence
galaxies whose intrinsic color is a tight function of redshift.  Each
crayon's color residual (observed minus the red-sequence locus at its
zredmagic) is a per-galaxy reddening probe, IMMUNE to the
magnification--color coupling that contaminated the QSO-color solve around
LRGs (ms.tex sec. 6): no flux-limit re-selection enters, because the
residual is measured per object rather than imposed by sample selection.

Estimator: the exact machinery of projects_desi/02_paircount.py (imported
by path), with DESI DR1 BGS foregrounds (0.05 < z < 0.45) and their LSS
randoms, both cut to the DES Y3 Gold footprint (redMaGiC joint mask,
NSIDE 4096 RING, FRACGOOD >= 0.5); crayons at 0.4 < zredmagic < 0.9 with
the same z_bg > z_fg + 0.2 buffer.  Colors {g-r, r-i, i-z, g-i} from
Y3 Gold SOF corrected (= dereddened) mags fetched by 00b; DECam
extinction ratios R = g 3.214, r 2.165, i 1.592, z 1.211.

WHERE TO RUN: astro4 or astrofs4/SLURM (catalogs on /home + shared git).
    python3 01_crayon_paircount.py                   # full run
    python3 01_crayon_paircount.py --max-fg 200000   # quick look
    python3 01_crayon_paircount.py --selftest        # locus-residual check
Output: $OUT/paircount_descrayon_qso.npz ("_qso" suffix kept so the 02/04
tooling reads it unchanged).
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
from astropy.io import fits

PC02_DEFAULT = os.path.expanduser(
    "~/github/projects_desi/programs_igmdust/02_paircount.py")
DES_DIR_DEFAULT = "/home/suzuki/data/DES/catalog/redmagic"
CRAYON_FILE = "y3a2_gold2.2.1_redmagic_highlum_highz.fits"
COLOR_FILE = "redmagic_highlum_highz_y3gold_colors.fits"
MASK_FILE = ("y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2"
             "_mask.fits")
MASK_NSIDE = 4096                     # RING (per the mask filename)
COLORS_DES = [("g", "r"), ("r", "i"), ("i", "z"), ("g", "i")]
R_I = 1.592                           # DECam i; g/r/z already in 02's table
CRAYON_ZRANGE = (0.40, 0.90)
LOCUS_DZ = 0.01                       # red-sequence locus bin width
LOCUS_NMIN = 200                      # min crayons per locus bin
FRACGOOD_MIN = 0.5
MAGERR_MAX = 0.5
JK_NSIDE = 16                         # finer than 02: overlap area is small


def load_pc(path):
    spec = importlib.util.spec_from_file_location("paircount02", path)
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    return pc


# ---------------------------------------------------------------- crayons ---
def locus_residuals(z, colors, valid):
    """Observed color minus the median red-sequence locus at that redshift."""
    zb = np.arange(CRAYON_ZRANGE[0], CRAYON_ZRANGE[1] + LOCUS_DZ, LOCUS_DZ)
    zc = 0.5 * (zb[:-1] + zb[1:])
    res = np.full_like(colors, np.nan)
    for c in range(colors.shape[1]):
        med = np.full(len(zc), np.nan)
        for i in range(len(zc)):
            m = valid[:, c] & (z >= zb[i]) & (z < zb[i + 1])
            if m.sum() >= LOCUS_NMIN:
                med[i] = np.median(colors[m, c])
        ok = np.isfinite(med)
        res[:, c] = colors[:, c] - np.interp(z, zc[ok], med[ok])
        r = res[valid[:, c], c]
        sig = 1.4826 * np.median(np.abs(r - np.median(r)))
        a, b = COLORS_DES[c]
        print(f"  locus {a}-{b}: {ok.sum()} z-bins, residual scatter "
              f"{sig:.3f} mag (robust)")
    return res


def load_crayons(des_dir):
    """redMaGiC crayons joined to Y3 Gold colors; locus residuals + valid."""
    with fits.open(f"{des_dir}/{CRAYON_FILE}") as h:
        d = h[1].data
        cid = np.asarray(d["coadd_object_id"], np.int64)
        ra = np.asarray(d["ra"], float)
        dec = np.asarray(d["dec"], float)
        z = np.asarray(d["zredmagic"], float)
    with fits.open(f"{des_dir}/{COLOR_FILE}") as h:
        g = h[1].data
        gid = np.asarray(g["coadd_object_id"], np.int64)
        mag = {b: np.asarray(g[f"sof_cm_mag_corrected_{b}"], float)
               for b in "griz"}
        err = {b: np.asarray(g[f"sof_cm_mag_err_{b}"], float) for b in "griz"}
        flags = np.asarray(g["flags_gold"])
        ext = np.asarray(g["extended_class_sof"])

    order = np.argsort(gid)
    pos = np.clip(np.searchsorted(gid[order], cid), 0, len(gid) - 1)
    idx = order[pos]
    has = gid[idx] == cid
    keep = (has & (z >= CRAYON_ZRANGE[0]) & (z <= CRAYON_ZRANGE[1])
            & (flags[idx] == 0) & (ext[idx] >= 2))
    idx = idx[keep]
    print(f"  crayons: {keep.sum():,} / {len(cid):,} "
          f"(color match + flags_gold=0 + ext>=2 + z in {CRAYON_ZRANGE})")

    bandok = {b: (np.isfinite(mag[b][idx]) & (mag[b][idx] > 10)
                  & (mag[b][idx] < 30) & (err[b][idx] < MAGERR_MAX))
              for b in "griz"}
    colors = np.column_stack([mag[a][idx] - mag[b][idx]
                              for a, b in COLORS_DES])
    valid = np.column_stack([bandok[a] & bandok[b] for a, b in COLORS_DES])
    print(f"  color validity: {valid.mean(0).round(3)}")

    res = locus_residuals(z[keep], colors, valid)
    valid &= np.isfinite(res)
    res = np.where(valid, res, 0.0)
    return ra[keep], dec[keep], z[keep], res, valid


# -------------------------------------------------------------- footprint ---
def ang2pix_ring(nside, ra_deg, dec_deg):
    """HEALPix RING ang2pix in pure numpy (healpy won't build on astro4;
    validated against healpy 1.20.1 to be bit-identical at nside 4096)."""
    z = np.sin(np.radians(dec_deg))
    tt = np.mod(np.radians(ra_deg), 2 * np.pi) / (np.pi / 2)      # [0, 4)
    pix = np.empty(len(z), dtype=np.int64)
    ncap = 2 * nside * (nside - 1)

    eq = np.abs(z) <= 2.0 / 3.0
    t1 = nside * (0.5 + tt[eq])
    t2 = nside * z[eq] * 0.75
    jp = np.floor(t1 - t2).astype(np.int64)      # ascending edge line
    jm = np.floor(t1 + t2).astype(np.int64)      # descending edge line
    ir = nside + 1 + jp - jm                     # ring counted from z=2/3
    kshift = 1 - (ir & 1)
    ip = (jp + jm - nside + kshift + 1) // 2
    ip = np.mod(ip, 4 * nside)
    pix[eq] = ncap + (ir - 1) * 4 * nside + ip

    po = ~eq
    tp = tt[po] - np.floor(tt[po])
    tmp = nside * np.sqrt(3.0 * (1.0 - np.abs(z[po])))
    jp = np.floor(tp * tmp).astype(np.int64)
    jm = np.floor((1.0 - tp) * tmp).astype(np.int64)
    ir = jp + jm + 1
    ip = np.mod(np.floor(tt[po] * ir).astype(np.int64), 4 * ir)
    north = z[po] > 0
    pix[po] = np.where(north, 2 * ir * (ir - 1) + ip,
                       12 * nside * nside - 2 * ir * (ir + 1) + ip)
    return pix


def load_footprint(des_dir):
    with fits.open(f"{des_dir}/{MASK_FILE}") as h:
        d = h[1].data
        pix = np.asarray(d["HPIX"], np.int64)
        frac = np.asarray(d["FRACGOOD"], float)
    good = np.sort(pix[frac >= FRACGOOD_MIN])
    print(f"  mask: {len(good):,} pixels with FRACGOOD >= {FRACGOOD_MIN} "
          f"(~{len(good) * 0.73 / 3600:,.0f} deg2)")
    return good


def in_footprint(good, ra, dec, label):
    p = ang2pix_ring(MASK_NSIDE, ra, dec)
    j = np.clip(np.searchsorted(good, p), 0, len(good) - 1)
    m = good[j] == p
    print(f"  {label}: {m.sum():,} / {len(ra):,} in DES footprint")
    return m


# ----------------------------------------------------------------- driver ---
def main(a):
    pc = load_pc(a.pc02)
    pc.R_COEF["i"] = R_I
    pc.COLORS = COLORS_DES
    pc.JK_NSIDE = JK_NSIDE
    pc.FG_ZRANGE["descrayon"] = pc.FG_ZRANGE["bgs"]
    # crayons are ~200/deg2 and low-z slices sweep 2.7 deg radii: 50k-fg
    # chunks OOM-killed the 20G SLURM job on the randoms pass; 10k is safe.
    pc.CHUNK = 10_000
    edges, rp_mid = pc.rp_bins()

    print("== crayons: DES redMaGiC highlum_highz + Y3 Gold colors")
    cra, cdec, cz, cres, cval = load_crayons(a.des_dir)
    good = load_footprint(a.des_dir)

    print("== foregrounds: DESI DR1 BGS")
    fra, fdec, fz, fw = pc.load_lss(a.base, "bgs", "dat")
    m = in_footprint(good, fra, fdec, "BGS")
    fra, fdec, fz, fw = fra[m], fdec[m], fz[m], fw[m]
    if m.sum() == 0:
        sys.exit("no BGS galaxies in the DES footprint — check mask/ordering")

    print("== randoms: DESI DR1 BGS LSS randoms")
    rra, rdec, rz, rw = pc.load_lss(a.base, "bgs", "ran", nran=a.nran)
    m = in_footprint(good, rra, rdec, "randoms")
    rra, rdec, rz, rw = rra[m], rdec[m], rz[m], rw[m]

    pc.run_arrays("descrayon", (fra, fdec, fz, fw), (rra, rdec, rz, rw),
                  (cra, cdec, cz, cres, cval), edges, rp_mid, a.out,
                  a.max_fg, tag=f"_sub{a.max_fg}" if a.max_fg else "")


# --------------------------------------------------------------- selftest ---
def selftest():
    """Locus-residual check: inject E(B-V)=0.05 into half the crayons.

    The mean residual difference between the reddened and clean halves
    must recover dR * 0.05 per color to <5%.  (The spatial estimator
    itself is validated by 02_paircount.py --selftest.)"""
    rng = np.random.default_rng(3)
    n = 200_000
    z = rng.uniform(*CRAYON_ZRANGE, n)
    dR = np.array([3.214 - 2.165, 2.165 - 1.592, 1.592 - 1.211,
                   3.214 - 1.592])
    locus = np.column_stack([1.2 + 0.8 * z - 0.3 * z ** 2, 0.9 + 0.5 * z,
                             0.45 + 0.2 * z, 2.1 + 1.1 * z])
    ebv = np.where(rng.random(n) < 0.5, 0.05, 0.0)
    colors = locus + rng.normal(0, 0.05, (n, 4)) + ebv[:, None] * dR
    valid = np.ones((n, 4), bool)
    res = locus_residuals(z, colors, valid)
    d = res[ebv > 0].mean(0) - res[ebv == 0].mean(0)
    frac = np.abs(d - dR * 0.05) / (dR * 0.05)
    for i, (a, b) in enumerate(COLORS_DES):
        print(f"  {a}-{b}: recovered {d[i]:+.5f}  "
              f"injected {0.05 * dR[i]:+.5f}  frac.dev {frac[i]:.2%}")
    if np.all(frac < 0.05):
        print("SELFTEST PASS")
        return 0
    print("SELFTEST FAIL")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pc02", default=PC02_DEFAULT,
                    help="path to projects_desi 02_paircount.py")
    ap.add_argument("--des-dir", default=DES_DIR_DEFAULT)
    ap.add_argument("--base", default=None,
                    help="DESI DR1 catalog dir (default: 02's)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: 02's)")
    ap.add_argument("--max-fg", type=int, default=0)
    ap.add_argument("--nran", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    pc_probe = load_pc(a.pc02)
    a.base = a.base or pc_probe.BASE_DEFAULT
    a.out = a.out or pc_probe.OUT_DEFAULT
    main(a)
