#!/usr/bin/env python3
"""
04_redmapper_crayons.py — igmdust/DES: dust around redMaPPer clusters,
measured with redMaGiC standard crayons (the McCleary massive-halo case).

Foregrounds: DES Y3 redMaPPer v6.4.22+2 clusters, the official lgt20
(lambda > 20) cosmology selection from the release file's index group
(19,388 clusters), with the release's own weighted randoms (ztrue).
Backgrounds: the redMaGiC highlum_highz standard crayons of 01 (locus
residuals about zredmagic; multiply profiles by 1/f_abs = 1/0.806 from
02_speczcalib when quoting de-absorbed reddening).

Estimator and bins identical to the other crayon runs (projects_desi 02
machinery; physical r_p at the cluster z_lambda; z buffer 0.2).

WHERE TO RUN: astrofs4/SLURM (10 cores, ~15 min) or astro4:
    sbatch -J igmdust_rm -d singleton 03_slurm_paircount.sbatch redmapper
Output: $OUT/paircount_redmapper_qso.npz
"""
import argparse
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "crayon01", os.path.join(_here, "01_crayon_paircount.py"))
m01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m01)

H5_DEFAULT = ("/home/suzuki/data/DES/catalog/redmapper/"
              "y3_redmapper_v6.4.22+2_release.h5")
INDEX = "index/redmapper/lgt20"      # official lambda>20 cosmology sample
FG_ZRANGE_RM = (0.10, 0.70)          # crayons are 0.4-0.9 with 0.2 buffer
MAX_RAN = 1_500_000                  # plenty: ~75x the cluster count


def load_clusters(h5path):
    import h5py
    with h5py.File(h5path, "r") as f:
        sel = f[f"{INDEX}/select"][:]
        g = f["catalog/cluster"]
        fra = g["ra"][:][sel]
        fdec = g["dec"][:][sel]
        fz = np.asarray(g["z_lambda"][:][sel], float)
        lam = g["lambda_chisq"][:][sel]
        print(f"  clusters (lgt20 select): {len(sel):,}  "
              f"z_lambda [{fz.min():.2f},{fz.max():.2f}]  "
              f"lambda [{lam.min():.0f},{lam.max():.0f}]")
        rsel = f[f"{INDEX}/random_select"][:]
        r = f["randoms"]
        rra = r["ra"][:][rsel]
        rdec = r["dec"][:][rsel]
        rz = np.asarray(r["ztrue"][:][rsel], float)
        rw = np.asarray(r["weight"][:][rsel], float)
    if len(rra) > MAX_RAN:
        rng = np.random.default_rng(3)
        pick = rng.choice(len(rra), MAX_RAN, replace=False)
        rra, rdec, rz, rw = rra[pick], rdec[pick], rz[pick], rw[pick]
    print(f"  randoms (release, weighted): {len(rra):,}")
    return ((fra, fdec, fz, np.ones(len(fra))), (rra, rdec, rz, rw))


def main(a):
    pc = m01.load_pc(a.pc02)
    pc.R_COEF["i"] = m01.R_I
    pc.COLORS = m01.COLORS_DES
    pc.JK_NSIDE = m01.JK_NSIDE
    pc.FG_ZRANGE["redmapper"] = FG_ZRANGE_RM
    pc.CHUNK = 10_000                # 20G SLURM limit (see 01)
    edges, rp_mid = pc.rp_bins()

    print("== crayons: DES redMaGiC highlum_highz + Y3 Gold colors")
    cra, cdec, cz, cres, cval = m01.load_crayons(a.des_dir)
    print("== foregrounds: DES Y3 redMaPPer clusters")
    fg, ran = load_clusters(a.h5)

    pc.run_arrays("redmapper", fg, ran, (cra, cdec, cz, cres, cval),
                  edges, rp_mid, a.out,
                  a.max_fg, tag=f"_sub{a.max_fg}" if a.max_fg else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--h5", default=H5_DEFAULT)
    ap.add_argument("--des-dir", default=m01.DES_DIR_DEFAULT)
    ap.add_argument("--pc02", default=m01.PC02_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-fg", type=int, default=0)
    a = ap.parse_args()
    a.out = a.out or m01.load_pc(a.pc02).OUT_DEFAULT
    main(a)
