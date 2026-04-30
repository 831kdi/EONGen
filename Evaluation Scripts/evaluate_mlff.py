"""
evaluate_mlff.py
MLFF-based energetic and stability metrics for nanocluster generation evaluation.

Metrics:
  1. Relaxation Stability (RMSD_relax):
       Each generated structure is relaxed with MACE-MP-0 MLFF.
       RMSD between initial and relaxed structure (Kabsch-aligned, no permutation).
       Small = already near a local minimum. Large = needed major rearrangement.

  2. Relaxation Hit Rate (H_0.5):
       Fraction of generated samples with RMSD_relax < 0.5 A.
       Reported per-target and overall.

  3. Basin Population–Energy Consistency (C_rank):
       For each composition (element + n_atoms), relax all K samples.
       Cluster relaxed structures into basins (DBSCAN, eps=--basin_eps).
       Compute Spearman rank correlation between basin population and basin energy.
       C_rank = -rho_s  (positive = low-energy basins are sampled more often).
       Skipped if fewer than 3 basins found.

Convergence criterion: max ||F_i|| < 0.05 eV/A, max 500 steps.
Non-converged structures are recorded as failures and excluded from C_rank.

Outputs:
  - mlff_raw_samples.csv       : per-sample RMSD_relax, energy, converged flag
  - mlff_basins.json           : per-target basin info (p_b, E_b, n_atoms, C_rank)
  - mlff_summary.json          : overall + size-bin + element stats
  Terminal: overall table + size-bin table + element-cluster table + hit rates

Usage:
  python evaluate_mlff.py \\
      --generated generated_all.xyz \\
      --target    target_all.xyz \\
      --corr_csv  correlations.csv \\
      --output    ./mlff_out \\
      --mace_device cuda          # or cpu
"""

import re
import json
import csv
import argparse
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist, squareform
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from ase import Atoms
from ase.data import chemical_symbols
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

FMAX_THRESHOLD  = 0.05   # eV/A
MAX_STEPS       = 500
HIT_THRESHOLD   = 0.5    # A — for H_0.5
MIN_BASINS_CRANK = 3     # skip C_rank if fewer basins

SIZE_BINS = [
    (1,  10,  "01-10"),
    (11, 20,  "11-20"),
    (21, 30,  "21-30"),
    (31, 40,  "31-40"),
    (41, 50,  "41-50"),
    (51, 999, "51+"),
]

SIZE_COLORS = {
    "01-10": "#e41a1c",
    "11-20": "#ff7f00",
    "21-30": "#4daf4a",
    "31-40": "#377eb8",
    "41-50": "#984ea3",
    "51+":   "#a65628",
}

CLUSTER_COLORS = {
    1: "#e41a1c",
    2: "#ff7f00",
    3: "#000000",
    4: "#4daf4a",
    5: "#377eb8",
    6: "#984ea3",
    7: "#a65628",
}
CLUSTER_LABELS = {
    1: "Pnictogen (Bi/Sb/As/P)",
    2: "Chalcogen+C (Te/C/Se/S/Mo)",
    3: "B",
    4: "Early TM+Alk.Earth",
    5: "Transition+sp metals",
    6: "V/Cr",
    7: "Refractory (Ta/Nb/Re/W)",
}

REPORT_KEYS = ["rmsd_relax", "energy", "converged_frac"]


# ─────────────────────────────────────────────────────────────
# Element helpers
# ─────────────────────────────────────────────────────────────

def to_symbol(val):
    try:
        z = int(val)
        return chemical_symbols[z]
    except (ValueError, TypeError):
        return str(val)


def build_element_clusters(corr_csv_path, k=7):
    if corr_csv_path is None or not Path(corr_csv_path).exists():
        return {}
    try:
        df  = pd.read_csv(corr_csv_path, index_col=0)
        dist = np.clip(1.0 - df.values, 0, None)
        np.fill_diagonal(dist, 0)
        Z   = linkage(squareform(dist), method='average')
        lbl = fcluster(Z, k, criterion='maxclust')
        return {el: int(l) for el, l in zip(df.index, lbl)}
    except Exception as e:
        print(f"  [warn] Could not build element clusters: {e}")
        return {}


def size_bin_label(n):
    for lo, hi, label in SIZE_BINS:
        if lo <= n <= hi:
            return label
    return "51+"


# ─────────────────────────────────────────────────────────────
# XYZ reader
# ─────────────────────────────────────────────────────────────

def read_xyz_file(filename):
    structures = []
    with open(filename, "r") as f:
        lines = f.readlines()
    i, n_lines = 0, len(lines)

    def _is_int(s):
        try:
            int(s.strip()); return True
        except Exception:
            return False

    while i < n_lines:
        while i < n_lines and lines[i].strip() == "":
            i += 1
        if i >= n_lines:
            break
        if not _is_int(lines[i]):
            i += 1; continue
        n_atoms = int(lines[i].strip())
        if n_atoms <= 0 or i + 1 >= n_lines:
            i += 1; continue
        comment = lines[i + 1].strip()
        start, end = i + 2, i + 2 + n_atoms
        if end > n_lines:
            break
        atoms_list, positions, ok = [], [], True
        for j in range(start, end):
            parts = lines[j].split()
            if len(parts) < 4:
                ok = False; break
            try:
                x, y, z_ = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                ok = False; break
            atoms_list.append(parts[0])
            positions.append([x, y, z_])
        if ok:
            structures.append({
                "atoms":     atoms_list,
                "positions": np.array(positions, dtype=float),
                "comment":   comment,
                "n_atoms":   n_atoms,
            })
            i = end
        else:
            i += 1
    return structures


# ─────────────────────────────────────────────────────────────
# MACE calculator
# ─────────────────────────────────────────────────────────────

_mace_calc = None

def init_mace(model_size="small", device="cpu", dtype="float32"):
    global _mace_calc
    try:
        from mace.calculators import mace_mp
        _mace_calc = mace_mp(
            model=model_size,
            device=device,
            default_dtype=dtype,
        )
        print(f"  MACE-MP-0 ({model_size}) loaded on {device}")
        return True
    except Exception as e:
        print(f"  [ERROR] Could not load MACE: {e}")
        print("  Install with: pip install mace-torch")
        return False


# ─────────────────────────────────────────────────────────────
# Geometry relaxation
# ─────────────────────────────────────────────────────────────

def relax_structure(atoms_list, positions, calc):
    """
    Relax a structure using the provided ASE calculator.
    Returns dict with:
        positions_relaxed  : np.array (N,3)
        energy             : float (eV), or None if failed
        max_force          : float (eV/A)
        converged          : bool
        n_steps            : int
    """
    from ase.optimize import FIRE as ASE_FIRE

    ase_atoms = Atoms(symbols=atoms_list, positions=positions.copy())
    ase_atoms.calc = calc

    try:
        # Initial energy sanity check
        e0 = ase_atoms.get_potential_energy()
        if not np.isfinite(e0):
            return _failed_relax()

        opt = ASE_FIRE(ase_atoms, logfile=None)
        opt.run(fmax=FMAX_THRESHOLD, steps=MAX_STEPS)

        pos_final = ase_atoms.get_positions()
        energy    = float(ase_atoms.get_potential_energy())
        forces    = ase_atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max())
        converged = max_force < FMAX_THRESHOLD
        n_steps   = opt.get_number_of_steps()

        if not np.isfinite(energy):
            return _failed_relax()

        return {
            "positions_relaxed": pos_final,
            "energy":            energy,
            "max_force":         max_force,
            "converged":         converged,
            "n_steps":           n_steps,
        }
    except Exception:
        return _failed_relax()


def _failed_relax():
    return {
        "positions_relaxed": None,
        "energy":            None,
        "max_force":         None,
        "converged":         False,
        "n_steps":           -1,
    }


# ─────────────────────────────────────────────────────────────
# Kabsch RMSD (no permutation — relaxation preserves atom order)
# ─────────────────────────────────────────────────────────────

def kabsch_rmsd(pos_init, pos_relaxed):
    """Kabsch-aligned RMSD between initial and relaxed positions."""
    if pos_relaxed is None:
        return np.nan
    if not (np.isfinite(pos_init).all() and np.isfinite(pos_relaxed).all()):
        return np.nan
    p = pos_init    - pos_init.mean(0)
    q = pos_relaxed - pos_relaxed.mean(0)
    try:
        U, S, Vt = np.linalg.svd(p.T @ q)
    except np.linalg.LinAlgError:
        return np.nan
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1., 1., d]) @ U.T
    rmsd = float(np.sqrt(np.mean(np.sum((p @ R.T - q) ** 2, axis=1))))
    return rmsd if np.isfinite(rmsd) else np.nan


# ─────────────────────────────────────────────────────────────
# Basin clustering + C_rank
# ─────────────────────────────────────────────────────────────

def pairwise_kabsch_rmsd(positions_list):
    """Compute full N×N pairwise RMSD matrix for a list of position arrays."""
    n = len(positions_list)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = kabsch_rmsd(positions_list[i], positions_list[j])
            mat[i, j] = mat[j, i] = (r if np.isfinite(r) else 999.0)
    return mat


def compute_basin_crank(relaxed_records, basin_eps=0.3):
    """
    Given a list of relaxed records (each has positions_relaxed + energy),
    cluster into basins and compute C_rank.

    Returns dict:
        basins      : list of {basin_id, n_members, population, energy, member_indices}
        C_rank      : float or None
        spearman_r  : float or None
        n_basins    : int
        skipped     : bool (True if too few basins)
    """
    # Filter to converged only
    valid = [(i, r) for i, r in enumerate(relaxed_records)
             if r["converged"] and r["positions_relaxed"] is not None
             and r["energy"] is not None and np.isfinite(r["energy"])]

    if len(valid) < 2:
        return {"basins": [], "C_rank": None, "spearman_r": None,
                "n_basins": 0, "skipped": True, "skip_reason": "fewer than 2 converged"}

    indices, records = zip(*valid)
    positions_list   = [r["positions_relaxed"] for r in records]
    energies         = np.array([r["energy"] for r in records])
    K                = len(relaxed_records)   # total samples for this target

    # Pairwise RMSD matrix
    rmsd_mat = pairwise_kabsch_rmsd(positions_list)

    # DBSCAN clustering
    db     = DBSCAN(eps=basin_eps, min_samples=1, metric="precomputed")
    labels = db.fit_predict(rmsd_mat)

    unique_basins = np.unique(labels)
    n_basins      = len(unique_basins)

    if n_basins < MIN_BASINS_CRANK:
        return {"basins": [], "C_rank": None, "spearman_r": None,
                "n_basins": n_basins, "skipped": True,
                "skip_reason": f"only {n_basins} basin(s), need >= {MIN_BASINS_CRANK}"}

    # Per-basin stats: representative energy = minimum energy member
    basin_list = []
    for b in unique_basins:
        mask         = labels == b
        n_members    = int(mask.sum())
        population   = n_members / K
        basin_energy = float(energies[mask].min())   # lowest energy in basin
        member_idx   = [int(indices[i]) for i, m in enumerate(mask) if m]
        basin_list.append({
            "basin_id":    int(b),
            "n_members":   n_members,
            "population":  population,
            "energy":      basin_energy,
            "member_indices": member_idx,
        })

    pops    = np.array([b["population"] for b in basin_list])
    e_vals  = np.array([b["energy"]     for b in basin_list])

    rho, pval  = spearmanr(pops, e_vals)
    C_rank     = float(-rho) if np.isfinite(rho) else None

    return {
        "basins":     basin_list,
        "C_rank":     C_rank,
        "spearman_r": float(rho) if np.isfinite(rho) else None,
        "n_basins":   n_basins,
        "skipped":    False,
        "skip_reason": None,
    }


# ─────────────────────────────────────────────────────────────
# Statistics & aggregation
# ─────────────────────────────────────────────────────────────

def stats(values):
    valid = [v for v in values if v is not None and np.isfinite(float(v))]
    if not valid:
        return {"mean": None, "std": None, "median": None, "n": 0}
    arr = np.array(valid)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "median": float(np.median(arr)), "n": len(valid)}


def aggregate(records, keys, key_fn=None):
    if key_fn is None:
        return {k: stats([r.get(k) for r in records]) for k in keys}
    groups = {}
    for r in records:
        g = key_fn(r)
        groups.setdefault(g, []).append(r)
    return {g: aggregate(recs, keys) for g, recs in
            sorted(groups.items(), key=lambda x: str(x[0]))}


def hit_rate(records, threshold=HIT_THRESHOLD):
    vals = [r["rmsd_relax"] for r in records
            if r.get("rmsd_relax") is not None and np.isfinite(r["rmsd_relax"])]
    if not vals:
        return None, 0
    rate = float(np.mean([v < threshold for v in vals]))
    return rate, len(vals)


# ─────────────────────────────────────────────────────────────
# Terminal printing
# ─────────────────────────────────────────────────────────────

def print_overall(overall, h_rate, n_total):
    W = 78
    print("\n" + "=" * W)
    print("  OVERALL SUMMARY  (all generated samples)")
    print("=" * W)
    for k in ["rmsd_relax", "energy"]:
        s = overall.get(k, {})
        if s.get("mean") is not None:
            print(f"  {k:<24}: {s['mean']:9.4f} ± {s['std']:8.4f}  "
                  f"(median {s['median']:9.4f},  n={s['n']})")
        else:
            print(f"  {k:<24}: N/A")
    c = overall.get("converged_frac", {})
    if c.get("mean") is not None:
        print(f"  {'convergence rate':<24}: {c['mean']*100:.1f}%  (n={c['n']})")
    if h_rate is not None:
        print(f"  {'hit rate H_0.5':<24}: {h_rate*100:.1f}%  (n={n_total})")
    print("=" * W)


def print_size_table(size_agg, size_hit_rates):
    W = 78
    print("\n" + "=" * W)
    print("  BY SIZE BIN")
    print("=" * W)
    print(f"  {'Bin':<8}  {'RMSD_relax':<22}  {'Energy (eV)':<22}  "
          f"{'Conv%':>6}  {'H_0.5%':>7}")
    print("-" * W)
    for bl, ms in size_agg.items():
        rmsd_s  = ms.get("rmsd_relax", {})
        e_s     = ms.get("energy", {})
        conv_s  = ms.get("converged_frac", {})
        rmsd_c  = (f"{rmsd_s['mean']:.3f}±{rmsd_s['std']:.3f}(n={rmsd_s['n']})"
                   if rmsd_s.get("mean") is not None else "N/A")
        e_c     = (f"{e_s['mean']:.3f}±{e_s['std']:.3f}"
                   if e_s.get("mean") is not None else "N/A")
        conv_c  = (f"{conv_s['mean']*100:.1f}" if conv_s.get("mean") is not None else "N/A")
        hr, _   = size_hit_rates.get(bl, (None, 0))
        hr_c    = (f"{hr*100:.1f}" if hr is not None else "N/A")
        print(f"  {bl:<8}  {rmsd_c:<22}  {e_c:<22}  {conv_c:>6}  {hr_c:>7}")
    print("=" * W)


def print_element_cluster_table(el_to_cluster, all_records, cluster_agg):
    W = 78
    print("\n" + "=" * W)
    print("  BY ELEMENT CLUSTER")
    print("=" * W)
    print(f"  {'Cluster':<35}  {'RMSD_relax':<18}  {'H_0.5%':>7}  {'C_rank':>8}")
    print("-" * W)
    for cid, ms in cluster_agg.items():
        label  = CLUSTER_LABELS.get(cid, f"Cluster {cid}")
        rmsd_s = ms.get("rmsd_relax", {})
        rmsd_c = (f"{rmsd_s['mean']:.3f}±{rmsd_s['std']:.3f}"
                  if rmsd_s.get("mean") is not None else "N/A")
        # hit rate for this cluster
        cl_recs = [r for r in all_records if el_to_cluster.get(r["element"]) == cid]
        hr, _   = hit_rate(cl_recs)
        hr_c    = (f"{hr*100:.1f}" if hr is not None else "N/A")
        print(f"  {label:<35}  {rmsd_c:<18}  {hr_c:>7}  {'—':>8}")
    print("=" * W)


def print_crank_summary(basin_results):
    valid = [(tidx, br) for tidx, br in basin_results.items()
             if not br["skipped"] and br["C_rank"] is not None]
    if not valid:
        print("\n  C_rank: no targets with >= 3 basins.")
        return
    cranks = [br["C_rank"] for _, br in valid]
    print(f"\n  C_rank (n={len(valid)} targets):  "
          f"mean={np.mean(cranks):.3f}  std={np.std(cranks):.3f}  "
          f"median={np.median(cranks):.3f}")
    print(f"  Fraction C_rank > 0: {np.mean([c > 0 for c in cranks])*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def generate_plots(all_records, basin_results, el_to_cluster, out_dir):
    # ── 1. RMSD_relax distribution violin ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Relaxation Stability (all samples)", fontsize=12, fontweight="bold")

    # Left: violin by size bin
    ax = axes[0]
    bin_data = {}
    for r in all_records:
        if r.get("rmsd_relax") is not None and np.isfinite(r["rmsd_relax"]):
            bl = size_bin_label(r["n_atoms"])
            bin_data.setdefault(bl, []).append(r["rmsd_relax"])

    bin_labels_sorted = [lbl for _, _, lbl in SIZE_BINS if lbl in bin_data]
    vdata = [bin_data[lbl] for lbl in bin_labels_sorted]
    if vdata:
        parts = ax.violinplot(vdata, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(SIZE_COLORS.get(bin_labels_sorted[i], "#888888"))
            pc.set_alpha(0.7)
        ax.axhline(HIT_THRESHOLD, color="red", linestyle="--", lw=1.2,
                   label=f"H_0.5 threshold ({HIT_THRESHOLD} Å)")
        ax.set_xticks(range(1, len(bin_labels_sorted) + 1))
        ax.set_xticklabels(bin_labels_sorted, rotation=45, fontsize=8)
        ax.set_ylabel("RMSD_relax (Å)")
        ax.set_title("By size bin")
        ax.legend(fontsize=8)

    # Right: violin by element cluster
    ax = axes[1]
    cl_data = {}
    for r in all_records:
        if r.get("rmsd_relax") is not None and np.isfinite(r["rmsd_relax"]):
            cl = el_to_cluster.get(r["element"])
            if cl is not None:
                cl_data.setdefault(cl, []).append(r["rmsd_relax"])

    cl_sorted = sorted(cl_data.keys())
    vdata2 = [cl_data[cl] for cl in cl_sorted]
    if vdata2:
        parts2 = ax.violinplot(vdata2, showmedians=True)
        for i, pc in enumerate(parts2["bodies"]):
            pc.set_facecolor(CLUSTER_COLORS.get(cl_sorted[i], "#888888"))
            pc.set_alpha(0.7)
        ax.axhline(HIT_THRESHOLD, color="red", linestyle="--", lw=1.2)
        ax.set_xticks(range(1, len(cl_sorted) + 1))
        ax.set_xticklabels([CLUSTER_LABELS.get(c, str(c)) for c in cl_sorted],
                           rotation=45, fontsize=6, ha="right")
        ax.set_ylabel("RMSD_relax (Å)")
        ax.set_title("By element cluster")

    plt.tight_layout()
    plt.savefig(out_dir / "mlff_rmsd_violin.png", dpi=150)
    plt.close()

    # ── 2. RMSD_relax vs n_atoms scatter ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("RMSD_relax vs Cluster Size", fontsize=12, fontweight="bold")
    for lo, hi, lbl in SIZE_BINS:
        color = SIZE_COLORS[lbl]
        xs = [r["n_atoms"]    for r in all_records
              if lo <= r["n_atoms"] <= hi
              and r.get("rmsd_relax") is not None and np.isfinite(r["rmsd_relax"])]
        ys = [r["rmsd_relax"] for r in all_records
              if lo <= r["n_atoms"] <= hi
              and r.get("rmsd_relax") is not None and np.isfinite(r["rmsd_relax"])]
        if xs:
            ax.scatter(xs, ys, c=color, alpha=0.25, s=12,
                       label=lbl, rasterized=True, linewidths=0)
    # Bin means
    all_n = np.array([r["n_atoms"]    for r in all_records
                      if r.get("rmsd_relax") is not None and np.isfinite(r["rmsd_relax"])])
    all_r = np.array([r["rmsd_relax"] for r in all_records
                      if r.get("rmsd_relax") is not None and np.isfinite(r["rmsd_relax"])])
    if len(all_n) > 0:
        bx = np.arange(all_n.min(), all_n.max() + 2, 5)
        bm, be = [], []
        for lo, hi in zip(bx[:-1], bx[1:]):
            mask = (all_n >= lo) & (all_n < hi)
            if mask.sum() > 0:
                be.append((lo + hi) / 2)
                bm.append(all_r[mask].mean())
        if be:
            ax.plot(be, bm, "k-o", ms=5, lw=2, label="bin mean", zorder=5)
    ax.axhline(HIT_THRESHOLD, color="red", linestyle="--", lw=1.2,
               label=f"H_0.5 threshold ({HIT_THRESHOLD} Å)")
    ax.set_xlabel("N atoms")
    ax.set_ylabel("RMSD_relax (Å)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "mlff_rmsd_vs_size.png", dpi=150)
    plt.close()

    # ── 3. Hit rate H_0.5 per element (bar) ──
    from collections import defaultdict
    el_hits = defaultdict(list)
    for r in all_records:
        if r.get("rmsd_relax") is not None and np.isfinite(r["rmsd_relax"]):
            el_hits[r["element"]].append(r["rmsd_relax"] < HIT_THRESHOLD)
    el_sorted = sorted(el_hits.keys())
    hr_vals   = [np.mean(el_hits[el]) * 100 for el in el_sorted]
    el_colors = [CLUSTER_COLORS.get(el_to_cluster.get(el), "#888888") for el in el_sorted]

    fig, ax = plt.subplots(figsize=(max(12, len(el_sorted) * 0.35), 5))
    ax.bar(range(len(el_sorted)), hr_vals, color=el_colors, alpha=0.8)
    ax.set_xticks(range(len(el_sorted)))
    ax.set_xticklabels(el_sorted, rotation=90, fontsize=7)
    ax.set_ylabel("Hit Rate H_0.5 (%)")
    ax.set_title("Relaxation Hit Rate per Element  (coloured by correlation cluster)",
                 fontsize=11, fontweight="bold")
    ax.axhline(50, color="red", linestyle="--", lw=1, label="50%")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "mlff_hitrate_by_element.png", dpi=150)
    plt.close()

    # ── 4. C_rank distribution ──
    cranks = [(tidx, br["C_rank"]) for tidx, br in basin_results.items()
              if not br["skipped"] and br["C_rank"] is not None]
    if cranks:
        fig, ax = plt.subplots(figsize=(8, 5))
        cvals = [c for _, c in cranks]
        ax.hist(cvals, bins=20, color="#377eb8", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", lw=1.5, label="C_rank=0")
        ax.axvline(np.median(cvals), color="orange", linestyle="-", lw=1.5,
                   label=f"median={np.median(cvals):.3f}")
        ax.set_xlabel("C_rank (−Spearman(population, energy))")
        ax.set_ylabel("Count (targets)")
        ax.set_title("Basin Population–Energy Consistency (C_rank)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "mlff_crank_hist.png", dpi=150)
        plt.close()

    print(f"  Plots saved to {out_dir}/mlff_*.png")


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def evaluate(
    generated_file,
    target_file,
    corr_csv=None,
    output_dir=None,
    mace_device="cpu",
    mace_model="small",
    mace_dtype="float32",
    basin_eps=0.3,
):
    out_dir = Path(output_dir) if output_dir else Path(generated_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build element clusters
    el_to_cluster = build_element_clusters(corr_csv)
    if el_to_cluster:
        print(f"  Element clusters loaded (k=7)")

    # Load MACE
    mace_ok = init_mace(model_size=mace_model, device=mace_device, dtype=mace_dtype)
    if not mace_ok:
        print("Aborting: MACE required for this evaluation.")
        return None

    # Load structures
    print(f"\nLoading generated: {generated_file}")
    generated = read_xyz_file(generated_file)
    print(f"  -> {len(generated)} structures")

    print(f"Loading targets:   {target_file}")
    targets_raw = read_xyz_file(target_file)
    print(f"  -> {len(targets_raw)} structures")

    # Parse targets
    tpat = re.compile(r"Target(?:\s+structure)?\s+(\d+)", re.IGNORECASE)
    epat = re.compile(r"element=(\w+)", re.IGNORECASE)
    targets_by_idx = {}
    for t in targets_raw:
        m = tpat.search(t["comment"])
        if not m:
            continue
        tidx = int(m.group(1))
        em   = epat.search(t["comment"])
        t["element"] = to_symbol(em.group(1)) if em else (t["atoms"][0] if t["atoms"] else "?")
        targets_by_idx[tidx] = t

    # Parse generated
    gpat = re.compile(r"\bTarget\s+(\d+)\b", re.IGNORECASE)
    spat = re.compile(r"\bsample\s+(\d+)\b", re.IGNORECASE)

    results_by_target = {}   # tidx -> {meta, samples: [...]}
    n_skip = 0

    print("\nRelaxing generated structures...")
    for gen in tqdm(generated, desc="Relaxing"):
        gm = gpat.search(gen["comment"])
        sm = spat.search(gen["comment"])
        if not gm:
            n_skip += 1; continue
        tidx = int(gm.group(1))
        sidx = int(sm.group(1)) if sm else -1

        if tidx not in targets_by_idx:
            n_skip += 1; continue
        tgt = targets_by_idx[tidx]
        if gen["n_atoms"] != tgt["n_atoms"]:
            n_skip += 1; continue

        pos_init = np.array(gen["positions"])
        if not (np.isfinite(pos_init).all() and np.abs(pos_init).max() < 1000):
            n_skip += 1; continue

        el = tgt["element"]

        # Relax
        relax_result = relax_structure(gen["atoms"], pos_init, _mace_calc)

        rmsd_r = kabsch_rmsd(pos_init, relax_result["positions_relaxed"])

        sample_rec = {
            "target_idx":       tidx,
            "sample_idx":       sidx,
            "element":          el,
            "n_atoms":          tgt["n_atoms"],
            "rmsd_relax":       float(rmsd_r) if np.isfinite(rmsd_r) else None,
            "energy":           relax_result["energy"],
            "max_force":        relax_result["max_force"],
            "converged":        relax_result["converged"],
            "converged_frac":   1.0 if relax_result["converged"] else 0.0,
            "n_steps":          relax_result["n_steps"],
            # Keep relaxed positions for basin clustering
            "_positions_relaxed": relax_result["positions_relaxed"],
        }

        if tidx not in results_by_target:
            results_by_target[tidx] = {
                "element":  el,
                "n_atoms":  tgt["n_atoms"],
                "comment":  tgt["comment"],
                "samples":  [],
            }
        results_by_target[tidx]["samples"].append(sample_rec)

    n_mapped_tgts = len(results_by_target)
    n_mapped_gen  = sum(len(v["samples"]) for v in results_by_target.values())
    print(f"\n  Mapped {n_mapped_tgts} targets, {n_mapped_gen} samples  ({n_skip} skipped)")

    # ── Basin clustering per target ────────────────────────────
    print("\nClustering basins per target...")
    basin_results = {}
    for tidx, data in tqdm(results_by_target.items(), desc="Basins"):
        # Build relaxed records list expected by compute_basin_crank
        relax_recs = [
            {
                "positions_relaxed": s["_positions_relaxed"],
                "energy":            s["energy"],
                "converged":         s["converged"],
            }
            for s in data["samples"]
        ]
        br = compute_basin_crank(relax_recs, basin_eps=basin_eps)
        basin_results[tidx] = br

    # ── Flat records (strip internal _positions_relaxed) ──────
    all_records = []
    for data in results_by_target.values():
        for s in data["samples"]:
            rec = {k: v for k, v in s.items() if not k.startswith("_")}
            all_records.append(rec)

    # ── Aggregation ───────────────────────────────────────────
    agg_keys     = ["rmsd_relax", "energy", "converged_frac"]
    overall_agg  = aggregate(all_records, agg_keys)
    size_bin_agg = aggregate(all_records, agg_keys,
                             key_fn=lambda r: size_bin_label(r["n_atoms"]))
    element_agg  = aggregate(all_records, agg_keys,
                             key_fn=lambda r: r["element"])
    cluster_agg  = aggregate(all_records, agg_keys,
                             key_fn=lambda r: el_to_cluster.get(r["element"], 0))

    # Hit rates
    h_overall, n_hr = hit_rate(all_records)
    size_hit_rates  = {
        bl: hit_rate([r for r in all_records if size_bin_label(r["n_atoms"]) == bl])
        for _, _, bl in SIZE_BINS
    }

    # ── Terminal output ────────────────────────────────────────
    print_overall(overall_agg, h_overall, n_hr)
    print_size_table(size_bin_agg, size_hit_rates)
    if el_to_cluster:
        print_element_cluster_table(el_to_cluster, all_records, cluster_agg)
    print_crank_summary(basin_results)

    # ── Plots ──────────────────────────────────────────────────
    print("\nGenerating plots...")
    generate_plots(all_records, basin_results, el_to_cluster, out_dir)

    # ── Save: mlff_raw_samples.csv ─────────────────────────────
    csv_fields = ["target_idx", "sample_idx", "element", "n_atoms",
                  "rmsd_relax", "energy", "max_force", "converged", "n_steps"]
    csv_path = out_dir / "mlff_raw_samples.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Saved mlff_raw_samples.csv    -> {csv_path}  ({len(all_records)} rows)")

    # ── Save: mlff_basins.json ─────────────────────────────────
    basin_out = {}
    for tidx, br in basin_results.items():
        data = results_by_target[tidx]
        basin_out[str(tidx)] = {
            "element":    data["element"],
            "n_atoms":    data["n_atoms"],
            "comment":    data["comment"],
            "n_basins":   br["n_basins"],
            "C_rank":     br["C_rank"],
            "spearman_r": br["spearman_r"],
            "skipped":    br["skipped"],
            "skip_reason":br.get("skip_reason"),
            "basins":     br["basins"],
        }
    basins_path = out_dir / "mlff_basins.json"
    with open(basins_path, "w") as f:
        json.dump(basin_out, f, indent=2)
    print(f"Saved mlff_basins.json        -> {basins_path}")

    # ── Save: mlff_summary.json ────────────────────────────────
    cranks_all = [br["C_rank"] for br in basin_results.values()
                  if not br["skipped"] and br["C_rank"] is not None]
    summary = {
        "n_targets":         n_mapped_tgts,
        "n_generated":       n_mapped_gen,
        "n_skipped":         n_skip,
        "hit_rate_H_0.5":    h_overall,
        "C_rank_mean":       float(np.mean(cranks_all)) if cranks_all else None,
        "C_rank_median":     float(np.median(cranks_all)) if cranks_all else None,
        "C_rank_n_targets":  len(cranks_all),
        "overall":           overall_agg,
        "by_size_bin":       size_bin_agg,
        "by_element":        element_agg,
    }
    sum_path = out_dir / "mlff_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved mlff_summary.json       -> {sum_path}")

    return summary


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="evaluate_mlff: MLFF relaxation-based stability metrics for nanoclusters"
    )
    parser.add_argument("--generated",   required=True,  help="Generated structures XYZ")
    parser.add_argument("--target",      required=True,  help="Target structures XYZ")
    parser.add_argument("--corr_csv",    default=None,   help="Element correlation CSV")
    parser.add_argument("--output",      default=None,   help="Output directory")
    parser.add_argument("--mace_device", default="cpu",  help="cpu or cuda")
    parser.add_argument("--mace_model",  default="small",
                        choices=["small", "medium", "large"],
                        help="MACE-MP-0 model size")
    parser.add_argument("--mace_dtype",  default="float32",
                        choices=["float32", "float64"])
    parser.add_argument("--basin_eps",   default=0.3, type=float,
                        help="DBSCAN epsilon (Å) for basin clustering (default: 0.3)")
    args = parser.parse_args()

    evaluate(
        generated_file = args.generated,
        target_file    = args.target,
        corr_csv       = args.corr_csv,
        output_dir     = args.output,
        mace_device    = args.mace_device,
        mace_model     = args.mace_model,
        mace_dtype     = args.mace_dtype,
        basin_eps      = args.basin_eps,
    )


if __name__ == "__main__":
    main()
