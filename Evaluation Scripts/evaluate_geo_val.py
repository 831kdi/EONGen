"""
Evaluate_geo_val.py
Geometric validity metrics for nanocluster generation evaluation.

Metrics (reference-free, applied to generated structures):
  - Steric Repulsion Score (Phi): LJ-inspired r^-12 penalty for atomic overlaps
  - Relative Shape Anisotropy (kappa^2): gyration tensor eigenvalue anisotropy
  - Radius of Gyration Scaling Consistency (delta_Rg): deviation from ideal
    compact sphere given bulk number density

Outlier detection: IQR method (Q3 + 1.5*IQR) applied globally across all
generated samples for each metric.

Outputs:
  - geo_raw_samples.csv          : per-sample raw values
  - geo_evaluation_detailed.json : per-target grouped results
  - geo_summary.json             : overall + size-bin + per-element stats
  Terminal: overall + size-bin table + outlier report
"""

import re
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from mendeleev import element as mendeleev_element
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

NA = 6.02214076e23   # Avogadro


# ─────────────────────────────────────────────────────────────
# Element helpers
# ─────────────────────────────────────────────────────────────

def to_symbol(val):
    try:
        z = int(val)
        return chemical_symbols[z]
    except (ValueError, TypeError):
        return str(val)


_density_cache = {}

def bulk_number_density(symbol):
    """
    Return bulk number density rho in atoms/A^3 using mendeleev experimental
    mass density and atomic weight.
    Returns None if data unavailable.
    """
    if symbol in _density_cache:
        return _density_cache[symbol]
    try:
        el = mendeleev_element(symbol)
        rho_gcc = el.density          # g/cm^3
        mw      = el.atomic_weight    # g/mol
        if rho_gcc is None or mw is None:
            _density_cache[symbol] = None
            return None
        rho = (rho_gcc * NA / mw) / 1e24   # atoms/A^3
        _density_cache[symbol] = rho
        return rho
    except Exception:
        _density_cache[symbol] = None
        return None


def cov_radius_angstrom(symbol):
    """Covalent radius in Angstrom from ASE (Alvarez 2008)."""
    try:
        z = atomic_numbers[symbol]
        r = covalent_radii[z]
        return float(r) if np.isfinite(r) else 1.50
    except Exception:
        return 1.50


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
        atoms, positions, ok = [], [], True
        for j in range(start, end):
            parts = lines[j].split()
            if len(parts) < 4:
                ok = False; break
            try:
                x, y, z_ = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                ok = False; break
            atoms.append(parts[0])
            positions.append([x, y, z_])
        if ok:
            structures.append({
                "atoms":     atoms,
                "positions": np.array(positions, dtype=float),
                "comment":   comment,
                "n_atoms":   n_atoms,
            })
            i = end
        else:
            i += 1
    return structures


# ─────────────────────────────────────────────────────────────
# Steric Repulsion Score (Phi)
# ─────────────────────────────────────────────────────────────

def steric_repulsion_score(atoms_list, positions):
    """
    Phi = 2/(N*(N-1)) * sum_{i<j} phi_ij
    phi_ij = (r_th_ij / r_ij)^12  if r_ij < r_th_ij, else 0
    r_th_ij = r_cov_i + r_cov_j
    """
    n = len(atoms_list)
    if n < 2:
        return 0.0
    try:
        radii = np.array([cov_radius_angstrom(s) for s in atoms_list])
        dm = cdist(positions, positions)   # (N, N)

        # threshold matrix r_th[i,j] = r_i + r_j
        r_th = radii[:, None] + radii[None, :]

        # mask upper triangle only (i < j), avoid diagonal
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        r_ij  = dm[mask]
        r_th_ij = r_th[mask]

        # only pairs where r_ij < r_th_ij get a penalty
        clash = r_ij < r_th_ij
        phi = np.zeros(len(r_ij))
        phi[clash] = (r_th_ij[clash] / r_ij[clash]) ** 12

        phi_total = phi.sum()
        n_pairs = n * (n - 1) / 2
        Phi = (phi_total / n_pairs)
        return float(Phi) if np.isfinite(Phi) else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────
# Relative Shape Anisotropy (kappa^2) via gyration tensor
# ─────────────────────────────────────────────────────────────

def relative_shape_anisotropy(positions):
    """
    Gyration tensor S = (1/N) * sum_i (r_i - r_CM)(r_i - r_CM)^T
    eigenvalues lambda_1, lambda_2, lambda_3
    lambda_bar = (l1 + l2 + l3) / 3

    kappa^2 = 3/2 * [(l1-lbar)^2 + (l2-lbar)^2 + (l3-lbar)^2] / (l1+l2+l3)^2
    kappa^2 in [0, 1]: 0 = sphere, 1 = rod
    """
    n = len(positions)
    if n < 2:
        return np.nan
    try:
        r = positions - positions.mean(axis=0)
        S = (r.T @ r) / n                           # (3,3) gyration tensor
        eigvals = np.linalg.eigvalsh(S)             # ascending order
        l1, l2, l3 = eigvals
        lbar = (l1 + l2 + l3) / 3.0
        denom = (l1 + l2 + l3) ** 2
        if denom < 1e-12:
            return np.nan
        kappa2 = 1.5 * ((l1 - lbar)**2 + (l2 - lbar)**2 + (l3 - lbar)**2) / denom
        return float(kappa2) if np.isfinite(kappa2) else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────
# Radius of Gyration Scaling Consistency (delta_Rg)
# ─────────────────────────────────────────────────────────────

def radius_of_gyration(positions):
    r = positions - positions.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum(r**2, axis=1))))
    return rg if np.isfinite(rg) else np.nan


def rg_ideal(n_atoms, element_symbol):
    """
    R_g_ideal = sqrt(3/5) * R  where R = (3N / 4*pi*rho)^(1/3)
    rho = bulk number density in atoms/A^3
    Returns (R_g_ideal, rho) or (None, None) if density unavailable.
    """
    rho = bulk_number_density(element_symbol)
    if rho is None or rho <= 0:
        return None, None
    R = (3 * n_atoms / (4 * np.pi * rho)) ** (1.0 / 3.0)
    Rg_id = np.sqrt(3.0 / 5.0) * R
    return float(Rg_id), float(rho)


def delta_rg(rg_obs, rg_id):
    """delta_Rg = (Rg - Rg_ideal) / Rg_ideal"""
    if rg_id is None or rg_id <= 0:
        return np.nan
    d = (rg_obs - rg_id) / rg_id
    return float(d) if np.isfinite(d) else np.nan


# ─────────────────────────────────────────────────────────────
# Per-sample metric computation
# ─────────────────────────────────────────────────────────────

def compute_metrics(gen, element_symbol):
    """
    All metrics computed on generated structure only (reference-free).
    element_symbol needed for delta_Rg bulk density lookup.
    """
    pos    = np.array(gen["positions"])
    atoms  = gen["atoms"]

    def _v(x):
        try:
            f = float(x)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    if not np.isfinite(pos).all() or np.abs(pos).max() > 1000:
        return dict(
            steric_repulsion=None,
            kappa2=None,
            rg=None, rg_ideal=None, delta_rg=None,
        )

    Phi    = _v(steric_repulsion_score(atoms, pos))
    kap2   = _v(relative_shape_anisotropy(pos))
    rg_obs = _v(radius_of_gyration(pos))
    rg_id, _ = rg_ideal(gen["n_atoms"], element_symbol)
    drg    = _v(delta_rg(rg_obs, rg_id)) if rg_obs is not None else None

    return dict(
        steric_repulsion = Phi,
        kappa2           = kap2,
        rg               = rg_obs,
        rg_ideal         = _v(rg_id),
        delta_rg         = drg,
    )


# ─────────────────────────────────────────────────────────────
# Statistics & aggregation
# ─────────────────────────────────────────────────────────────

REPORT_KEYS = ["steric_repulsion", "kappa2", "delta_rg"]


def stats(values):
    valid = [v for v in values if v is not None and np.isfinite(float(v))]
    if not valid:
        return {"mean": None, "std": None, "median": None, "n": 0}
    arr = np.array(valid)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "median": float(np.median(arr)),
        "n":      len(valid),
    }


def aggregate(records, key_fn=None):
    if key_fn is None:
        return {k: stats([r.get(k) for r in records]) for k in REPORT_KEYS}
    groups = {}
    for r in records:
        g = key_fn(r)
        groups.setdefault(g, []).append(r)
    return {g: aggregate(recs) for g, recs in sorted(groups.items(), key=lambda x: str(x[0]))}


def size_bin(n):
    if n <= 0:
        return "unknown"
    low  = ((n - 1) // 10) * 10 + 1
    high = low + 9
    return f"{low:02d}-{high:02d}"


# ─────────────────────────────────────────────────────────────
# IQR-based outlier detection
# ─────────────────────────────────────────────────────────────

def iqr_outliers(records, metric_key, k=1.5):
    """
    Returns list of records whose metric_key value exceeds Q3 + k*IQR.
    For delta_rg we also flag below Q1 - k*IQR (over-compressed).
    """
    vals = np.array([r[metric_key] for r in records
                     if r.get(metric_key) is not None and np.isfinite(r[metric_key])])
    if len(vals) < 4:
        return [], None, None

    Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    IQR = Q3 - Q1
    upper = Q3 + k * IQR
    lower = Q1 - k * IQR   # used for delta_rg (negative = over-compressed)

    outliers = []
    for r in records:
        v = r.get(metric_key)
        if v is None or not np.isfinite(v):
            continue
        if metric_key == "delta_rg":
            if v > upper or v < lower:
                outliers.append((r, v))
        else:
            if v > upper:
                outliers.append((r, v))

    return outliers, float(upper), float(lower)


# ─────────────────────────────────────────────────────────────
# Terminal printing
# ─────────────────────────────────────────────────────────────

def _fmt(v):
    """Auto-format: scientific if |v| > 1e4 or very small, else fixed 4dp."""
    if v is None:
        return "N/A"
    if abs(v) > 1e4 or (v != 0 and abs(v) < 1e-4):
        return f"{v:.3e}"
    return f"{v:.4f}"


def print_overall(agg):
    print("\n" + "=" * 72)
    print("  OVERALL SUMMARY  (all generated samples)")
    print("=" * 72)
    for k in REPORT_KEYS:
        s = agg.get(k, {})
        if s.get("mean") is not None:
            print(f"  {k:<24}: {_fmt(s['mean']):>14} ± {_fmt(s['std']):>14}  "
                  f"(median {_fmt(s['median']):>14},  n={s['n']})")
        else:
            print(f"  {k:<24}: N/A")
    print("=" * 72)


def print_size_table(size_agg):
    col_w = 26
    print("\n" + "=" * 72)
    print("  BY SIZE BIN  (all generated samples)")
    print("=" * 72)
    print(f"  {'Bin':<10}" + "".join(f"  {k:<{col_w}}" for k in REPORT_KEYS))
    print("-" * 72)
    for bin_label, mstats in size_agg.items():
        row = f"  {bin_label:<10}"
        for k in REPORT_KEYS:
            s = mstats.get(k, {})
            if s.get("mean") is not None:
                m, sd = s["mean"], s["std"]
                if abs(m) > 1e4 or abs(sd) > 1e4:
                    cell = f"{m:.2e}±{sd:.1e}(n={s['n']})"
                else:
                    cell = f"{m:.3f}±{sd:.3f}(n={s['n']})"
            else:
                cell = "N/A"
            row += f"  {cell:<{col_w}}"
        print(row)
    print("=" * 72)


def print_outliers(outliers_dict):
    print("\n" + "=" * 72)
    print("  OUTLIER REPORT  (IQR method, k=1.5)")
    print("=" * 72)
    for metric_key, (outliers, upper, lower) in outliers_dict.items():
        if metric_key == "delta_rg":
            print(f"\n  [{metric_key}]  threshold: < {lower:.4f} or > {upper:.4f}  "
                  f"({len(outliers)} outliers)")
        else:
            print(f"\n  [{metric_key}]  threshold: > {upper:.4f}  "
                  f"({len(outliers)} outliers)")
        if outliers:
            print(f"  {'target_idx':>10}  {'sample_idx':>10}  {'element':>8}  "
                  f"{'n_atoms':>8}  {'value':>12}")
            print("  " + "-" * 56)
            for r, v in sorted(outliers, key=lambda x: abs(x[1]), reverse=True)[:30]:
                print(f"  {r.get('target_idx','?'):>10}  {r.get('sample_idx','?'):>10}  "
                      f"{r.get('element','?'):>8}  {r.get('n_atoms','?'):>8}  {v:>12.4f}")
            if len(outliers) > 30:
                print(f"  ... ({len(outliers) - 30} more not shown)")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def evaluate(generated_file, target_file, output_dir=None):
    out_dir = Path(output_dir) if output_dir else Path(generated_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading generated: {generated_file}")
    generated = read_xyz_file(generated_file)
    print(f"  -> {len(generated)} structures")

    print(f"Loading targets:   {target_file}")
    targets = read_xyz_file(target_file)
    print(f"  -> {len(targets)} structures")

    # Parse targets: index -> element symbol
    tpat = re.compile(r"Target(?:\s+structure)?\s+(\d+)", re.IGNORECASE)
    epat = re.compile(r"element=(\w+)", re.IGNORECASE)
    targets_by_idx = {}
    for t in targets:
        m = tpat.search(t["comment"])
        if not m:
            continue
        tidx = int(m.group(1))
        em   = epat.search(t["comment"])
        element_sym = to_symbol(em.group(1)) if em else (t["atoms"][0] if t["atoms"] else "?")
        t["element"] = element_sym
        targets_by_idx[tidx] = t

    # Parse generated & evaluate
    gpat = re.compile(r"\bTarget\s+(\d+)\b", re.IGNORECASE)
    spat = re.compile(r"\bsample\s+(\d+)\b", re.IGNORECASE)

    results_by_target = {}
    n_skip = 0

    for gen in tqdm(generated, desc="Evaluating"):
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

        element_sym = tgt["element"]
        m = compute_metrics(gen, element_sym)
        m["target_idx"] = tidx
        m["sample_idx"] = sidx
        m["element"]    = element_sym
        m["n_atoms"]    = tgt["n_atoms"]

        if tidx not in results_by_target:
            results_by_target[tidx] = {
                "element":            element_sym,
                "n_atoms":            tgt["n_atoms"],
                "target_comment":     tgt["comment"],
                "metrics_per_sample": [],
            }
        results_by_target[tidx]["metrics_per_sample"].append(m)

    n_mapped_tgts = len(results_by_target)
    n_mapped_gen  = sum(len(v["metrics_per_sample"]) for v in results_by_target.values())
    print(f"\n  Mapped {n_mapped_tgts} targets, {n_mapped_gen} samples  ({n_skip} skipped)")

    # Flat list of all sample records
    all_records = []
    for data in results_by_target.values():
        all_records.extend(data["metrics_per_sample"])

    # ── Aggregations (over ALL samples) ──
    overall_agg  = aggregate(all_records)
    size_bin_agg = aggregate(all_records, key_fn=lambda r: size_bin(r["n_atoms"]))
    element_agg  = aggregate(all_records, key_fn=lambda r: r["element"])

    # ── Outlier detection ──
    outliers_dict = {}
    for mk in REPORT_KEYS:
        outliers, upper, lower = iqr_outliers(all_records, mk, k=1.5)
        outliers_dict[mk] = (outliers, upper, lower)

    # ── Terminal output ──
    print_overall(overall_agg)
    print_size_table(size_bin_agg)
    print_outliers(outliers_dict)

    # ── Save: geo_raw_samples.csv ──
    csv_path = out_dir / "geo_raw_samples.csv"
    csv_fields = [
        "target_idx", "sample_idx", "element", "n_atoms",
        "steric_repulsion", "kappa2",
        "rg", "rg_ideal", "delta_rg",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nSaved geo_raw_samples.csv          -> {csv_path}  ({len(all_records)} rows)")

    # ── Save: geo_evaluation_detailed.json ──
    detailed = {}
    for tidx, data in results_by_target.items():
        # best sample = lowest steric_repulsion
        sr_vals = [(s["steric_repulsion"] if s["steric_repulsion"] is not None else np.inf)
                   for s in data["metrics_per_sample"]]
        best_idx = int(np.argmin(sr_vals)) if not all(np.isinf(sr_vals)) else -1
        detailed[str(tidx)] = {
            "element":            data["element"],
            "n_atoms":            data["n_atoms"],
            "target_comment":     data["target_comment"],
            "best_sample_idx":    best_idx,
            "metrics_per_sample": data["metrics_per_sample"],
        }
    det_path = out_dir / "geo_evaluation_detailed.json"
    with open(det_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"Saved geo_evaluation_detailed.json -> {det_path}")

    # ── Save: geo_summary.json ──
    # Outlier summary (serialisable)
    outlier_summary = {}
    for mk, (outliers, upper, lower) in outliers_dict.items():
        outlier_summary[mk] = {
            "n_outliers": len(outliers),
            "upper_threshold": upper,
            "lower_threshold": lower,
            "outlier_records": [
                {"target_idx": r["target_idx"], "sample_idx": r["sample_idx"],
                 "element": r["element"], "n_atoms": r["n_atoms"], "value": v}
                for r, v in outliers
            ],
        }

    # ── Robust stats: mean & median after removing outliers ──
    robust_stats = {}
    for mk in REPORT_KEYS:
        outlier_records_set = {id(r) for r, _ in outliers_dict[mk][0]}
        clean_vals = [
            r[mk] for r in all_records
            if id(r) not in outlier_records_set
            and r.get(mk) is not None
            and np.isfinite(r[mk])
        ]
        if clean_vals:
            arr = np.array(clean_vals)
            robust_stats[mk] = {
                "mean":   float(arr.mean()),
                "median": float(np.median(arr)),
                "std":    float(arr.std()),
                "n":      len(arr),
                "n_removed": len(outliers_dict[mk][0]),
            }
        else:
            robust_stats[mk] = {
                "mean": None, "median": None, "std": None,
                "n": 0, "n_removed": len(outliers_dict[mk][0]),
            }

    summary = {
        "n_targets":    n_mapped_tgts,
        "n_generated":  n_mapped_gen,
        "n_skipped":    n_skip,
        "overall":      overall_agg,
        "by_size_bin":  size_bin_agg,
        "by_element":   element_agg,
        "outliers":     outlier_summary,
        "robust_stats": robust_stats,
    }
    sum_path = out_dir / "geo_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved geo_summary.json             -> {sum_path}")

    generate_plots_geo_val(all_records, out_dir)

    return summary


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def generate_plots_geo_val(records, out_dir):
    import warnings
    warnings.filterwarnings("ignore")
    from collections import defaultdict

    metrics_labels = {
        "steric_repulsion": "Steric Repulsion (Phi)",
        "kappa2":           "Relative Shape Anisotropy (kappa^2)",
        "delta_rg":         "RoG Scaling Deviation (delta_Rg)",
    }

    # ── 1. Overall distributions (violin, log scale for steric) ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Geometric Validity Distributions (all samples)", fontsize=12, fontweight="bold")
    for ax, (mk, label) in zip(axes, metrics_labels.items()):
        vals = [r[mk] for r in records if r.get(mk) is not None and np.isfinite(r[mk])]
        if vals:
            # clip steric repulsion for visualisation
            if mk == "steric_repulsion":
                vals = np.clip(vals, 0, np.percentile(vals, 99))
            ax.violinplot(vals, showmedians=True)
            ax.set_title(label, fontsize=9)
            ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(out_dir / "geo_distributions.png", dpi=150)
    plt.close()

    # ── 2. Metric vs n_atoms ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Geometric Validity vs Cluster Size", fontsize=12, fontweight="bold")
    for ax, (mk, label) in zip(axes, metrics_labels.items()):
        xs_raw = [r["n_atoms"] for r in records if r.get(mk) is not None and np.isfinite(r[mk])]
        ys_raw = [r[mk]        for r in records if r.get(mk) is not None and np.isfinite(r[mk])]
        if not xs_raw:
            continue
        xs, ys = np.array(xs_raw), np.array(ys_raw)
        # clip steric for vis
        if mk == "steric_repulsion":
            clip_val = np.percentile(ys, 99)
            ys = np.clip(ys, 0, clip_val)
        ax.scatter(xs, ys, alpha=0.15, s=8, color="steelblue", rasterized=True)
        bins = np.arange(xs.min(), xs.max() + 2, 5)
        bin_edges, bin_means = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (xs >= lo) & (xs < hi)
            if mask.sum() > 0:
                bin_edges.append((lo + hi) / 2)
                bin_means.append(ys[mask].mean())
        if bin_edges:
            ax.plot(bin_edges, bin_means, "r-o", ms=4, lw=2, label="bin mean")
            ax.legend(fontsize=8)
        ax.set_xlabel("N atoms")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "geo_vs_size.png", dpi=150)
    plt.close()

    # ── 3. Per-element bar chart ──
    elem_data = defaultdict(lambda: {mk: [] for mk in metrics_labels})
    for r in records:
        el = r.get("element", "?")
        for mk in metrics_labels:
            v = r.get(mk)
            if v is not None and np.isfinite(v):
                elem_data[el][mk].append(v)

    elements_sorted = sorted(elem_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(elements_sorted) * 0.35), 5))
    fig.suptitle("Per-Element Mean ± Std (all samples)", fontsize=12, fontweight="bold")
    for ax, (mk, label) in zip(axes, metrics_labels.items()):
        means, stds = [], []
        for el in elements_sorted:
            v = elem_data[el][mk]
            if v:
                if mk == "steric_repulsion":
                    clipped = np.clip(v, 0, np.percentile(v, 95))
                    means.append(np.mean(clipped))
                    stds.append(np.std(clipped))
                else:
                    means.append(np.mean(v))
                    stds.append(np.std(v))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        xs = np.arange(len(elements_sorted))
        ax.bar(xs, means, yerr=stds, capsize=2, color="coral", alpha=0.75, error_kw={"elinewidth": 0.8})
        ax.set_xticks(xs)
        ax.set_xticklabels(elements_sorted, rotation=90, fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "geo_by_element.png", dpi=150)
    plt.close()

    print(f"Saved plots -> {out_dir}/geo_*.png")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="evaluate_geo_val: geometric validity metrics for nanoclusters"
    )
    parser.add_argument("--generated", required=True, help="Generated structures XYZ")
    parser.add_argument("--target",    required=True, help="Target structures XYZ")
    parser.add_argument("--output",    default=None,  help="Output directory")
    args = parser.parse_args()
    evaluate(args.generated, args.target, args.output)


if __name__ == "__main__":
    main()
