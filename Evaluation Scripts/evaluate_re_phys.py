"""
Evaluate_re_phys.py
Reconstruction + Physical metrics for nanocluster generation evaluation.

Metrics:
  - Hungarian RMSD (Kabsch-aligned, permutation-invariant)
  - Radius of Gyration (gen, target, diff)
  - RDF Wasserstein distance (gen vs target)
  - CN Wasserstein distance (gen vs target, each with adaptive RDF-based cutoff)

Outputs:
  - raw_samples.csv          : per-sample raw values (all samples)
  - evaluation_detailed.json : per-target grouped results with best_sample_idx
  - summary_overall.json     : overall + size-bin + per-element stats
  Terminal: overall + size-bin + per-element summary table
  Plots saved to output directory
"""

import re
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from scipy.signal import argrelmin, argrelmax
from ase.data import chemical_symbols
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# Element helper
# ─────────────────────────────────────────────────────────────

def to_symbol(val):
    """Convert atomic number (int or str) or symbol string to element symbol."""
    try:
        z = int(val)
        return chemical_symbols[z]
    except (ValueError, TypeError):
        return str(val)


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
            int(s.strip())
            return True
        except Exception:
            return False

    while i < n_lines:
        while i < n_lines and lines[i].strip() == "":
            i += 1
        if i >= n_lines:
            break
        if not _is_int(lines[i]):
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        if n_atoms <= 0 or i + 1 >= n_lines:
            i += 1
            continue
        comment = lines[i + 1].strip()
        start, end = i + 2, i + 2 + n_atoms
        if end > n_lines:
            break
        atoms, positions, ok = [], [], True
        for j in range(start, end):
            parts = lines[j].split()
            if len(parts) < 4:
                ok = False
                break
            try:
                x, y, z_ = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                ok = False
                break
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
# Kabsch alignment + Hungarian RMSD
# ─────────────────────────────────────────────────────────────

def kabsch_align(pos1, pos2):
    """Kabsch-align pos1 onto pos2. Returns (aligned_pos1, rmsd)."""
    if not (np.isfinite(pos1).all() and np.isfinite(pos2).all()):
        return pos1, np.nan
    p = pos1 - pos1.mean(0)
    q = pos2 - pos2.mean(0)
    try:
        U, _, Vt = np.linalg.svd(p.T @ q)
    except np.linalg.LinAlgError:
        return pos1, np.nan
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1., 1., d]) @ U.T
    p_rot = p @ R.T
    rmsd = float(np.sqrt(np.mean(np.sum((p_rot - q) ** 2, axis=1))))
    return p_rot + pos2.mean(0), (rmsd if np.isfinite(rmsd) else np.nan)


def hungarian_rmsd(pos1, pos2):
    """Permutation-invariant Kabsch RMSD via Hungarian matching."""
    if not (np.isfinite(pos1).all() and np.isfinite(pos2).all()):
        return np.nan
    try:
        D = cdist(pos1, pos2)
        row_ind, _ = linear_sum_assignment(D)
        _, rmsd = kabsch_align(pos1[row_ind], pos2)
        return rmsd
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────
# Radius of Gyration
# ─────────────────────────────────────────────────────────────

def radius_of_gyration(positions):
    if not np.isfinite(positions).all():
        return np.nan
    r = positions - positions.mean(0)
    rog = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
    return rog if np.isfinite(rog) else np.nan


# ─────────────────────────────────────────────────────────────
# RDF + adaptive cutoff
# ─────────────────────────────────────────────────────────────

def compute_rdf(positions, r_max=8.0, n_bins=200):
    """Return (counts, bin_centers)."""
    bins = np.linspace(0, r_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    if len(positions) < 2:
        return np.zeros(n_bins), centers
    dm = cdist(positions, positions)
    np.fill_diagonal(dm, np.nan)
    dists = dm[np.isfinite(dm) & (dm < r_max)]
    counts, _ = np.histogram(dists, bins=bins)
    return counts.astype(float), centers


def rdf_adaptive_cutoff(positions, r_max=8.0, n_bins=200, smooth_window=7):
    """
    Find cutoff = first minimum of RDF after the first peak.
    Falls back to 1.3x first-peak position if no minimum found.
    Falls back to 3.5 A if no peak found.
    """
    counts, centers = compute_rdf(positions, r_max, n_bins)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        counts_s = np.convolve(counts, kernel, mode='same')
    else:
        counts_s = counts

    peaks = argrelmax(counts_s, order=3)[0]
    if len(peaks) == 0:
        return 3.5

    first_peak_idx = peaks[0]
    tail = counts_s[first_peak_idx:]
    mins_after = argrelmin(tail, order=3)[0]
    if len(mins_after) == 0:
        return float(centers[first_peak_idx]) * 1.3

    cutoff_idx = first_peak_idx + mins_after[0]
    return float(centers[cutoff_idx])


def rdf_wasserstein_dist(pos1, pos2, r_max=8.0, n_bins=200):
    try:
        c1, edges = compute_rdf(pos1, r_max, n_bins)
        c2, _     = compute_rdf(pos2, r_max, n_bins)
        s1, s2 = c1.sum(), c2.sum()
        if s1 == 0 or s2 == 0:
            return np.nan
        return float(wasserstein_distance(edges, edges, c1 / s1, c2 / s2))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────
# Coordination Number
# ─────────────────────────────────────────────────────────────

from ase.data import covalent_radii, atomic_numbers

def get_covalent_cutoff(symbols_i, symbols_j, scale=1.2):
    """
    Pairwise cutoff = scale * (r_cov_i + r_cov_j).
    symbols_i, symbols_j: element symbol strings.
    """
    ri = covalent_radii[atomic_numbers[symbols_i]]
    rj = covalent_radii[atomic_numbers[symbols_j]]
    return scale * (ri + rj)


def compute_cn_covalent(positions, symbols, scale=1.2):
    """
    Compute per-atom CN using element-specific covalent radii cutoffs.
    positions: (N, 3) array
    symbols: list of N element symbol strings
    Returns: (N,) float array of coordination numbers
    """
    n = len(positions)
    if n < 2:
        return np.zeros(n)
    dm = cdist(positions, positions)
    cn = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cutoff = get_covalent_cutoff(symbols[i], symbols[j], scale)
            if dm[i, j] < cutoff:
                cn[i] += 1
    return cn


def cn_wasserstein_dist(pos_gen, pos_tgt, symbols_gen, symbols_tgt, scale=1.2):
    """
    CN with element-specific covalent radii cutoffs, then Wasserstein between distributions.
    Returns (wasserstein_dist, cutoff_placeholder_gen, cutoff_placeholder_tgt).
    The cutoff values returned are the mean pairwise cutoff for each structure.
    """
    try:
        cn_gen = compute_cn_covalent(pos_gen, symbols_gen, scale)
        cn_tgt = compute_cn_covalent(pos_tgt, symbols_tgt, scale)
        if len(cn_gen) == 0 or len(cn_tgt) == 0:
            return np.nan, np.nan, np.nan
        # For logging: mean cutoff = scale * 2 * mean covalent radius
        mean_cut_gen = scale * 2 * np.mean([covalent_radii[atomic_numbers[s]] for s in symbols_gen])
        mean_cut_tgt = scale * 2 * np.mean([covalent_radii[atomic_numbers[s]] for s in symbols_tgt])
        return float(wasserstein_distance(cn_gen, cn_tgt)), mean_cut_gen, mean_cut_tgt
    except Exception:
        return np.nan, np.nan, np.nan


# ─────────────────────────────────────────────────────────────
# Per-sample metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(gen, tgt):
    gpos = np.array(gen["positions"])
    tpos = np.array(tgt["positions"])

    def _v(x):
        try:
            f = float(x)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    null = dict(
        hungarian_rmsd=None, rog_gen=None, rog_tgt=None, rog_diff=None,
        rdf_wasserstein=None, cn_wasserstein=None,
        cn_cutoff_gen=None, cn_cutoff_tgt=None,
    )

    if not np.isfinite(gpos).all() or np.abs(gpos).max() > 1000:
        return null

    gsyms = [to_symbol(a) for a in gen["atoms"]]
    tsyms = [to_symbol(a) for a in tgt["atoms"]]

    h_rmsd = _v(hungarian_rmsd(gpos, tpos))
    rog_g  = _v(radius_of_gyration(gpos))
    rog_t  = _v(radius_of_gyration(tpos))
    rog_d  = _v(abs(rog_g - rog_t)) if (rog_g is not None and rog_t is not None) else None
    rdf_w  = _v(rdf_wasserstein_dist(gpos, tpos))
    cn_w, cut_g, cut_t = cn_wasserstein_dist(gpos, tpos, gsyms, tsyms)

    return dict(
        hungarian_rmsd  = h_rmsd,
        rog_gen         = rog_g,
        rog_tgt         = rog_t,
        rog_diff        = rog_d,
        rdf_wasserstein = rdf_w,
        cn_wasserstein  = _v(cn_w),
        cn_cutoff_gen   = _v(cut_g),
        cn_cutoff_tgt   = _v(cut_t),
    )


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

REPORT_KEYS = ["hungarian_rmsd", "rog_diff", "rdf_wasserstein", "cn_wasserstein"]


def stats(values):
    valid = [v for v in values if v is not None and np.isfinite(float(v))]
    if not valid:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {
        "mean":   float(np.mean(valid)),
        "std":    float(np.std(valid)),
        "median": float(np.median(valid)),
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
    low = ((n - 1) // 10) * 10 + 1
    high = low + 9
    return f"{low:02d}-{high:02d}"


# ─────────────────────────────────────────────────────────────
# Terminal printing
# ─────────────────────────────────────────────────────────────

def print_overall(agg):
    print("\n" + "=" * 75)
    print("  OVERALL SUMMARY  (all generated samples)")
    print("=" * 75)
    for k in REPORT_KEYS:
        s = agg.get(k, {})
        if s.get("mean") is not None:
            print(f"  {k:<28}: {s['mean']:8.4f} ± {s['std']:7.4f}  "
                  f"(median {s['median']:8.4f},  n={s['n']})")
        else:
            print(f"  {k:<28}: N/A")
    print("=" * 75)


def print_size_table(size_agg):
    col_w = 24
    print("\n" + "=" * 75)
    print("  BY SIZE BIN  (all generated samples)")
    print("=" * 75)
    print(f"  {'Bin':<10}" + "".join(f"  {k:<{col_w}}" for k in REPORT_KEYS))
    print("-" * 75)
    for bin_label, metric_stats in size_agg.items():
        row = f"  {bin_label:<10}"
        for k in REPORT_KEYS:
            s = metric_stats.get(k, {})
            if s.get("mean") is not None:
                cell = f"{s['mean']:.3f}±{s['std']:.3f}(n={s['n']})"
            else:
                cell = "N/A"
            row += f"  {cell:<{col_w}}"
        print(row)
    print("=" * 75)



def print_element_table(element_agg):
    col_w = 24
    print("\n" + "=" * 75)
    print("  BY ELEMENT  (all generated samples)")
    print("=" * 75)
    print(f"  {'Element':<10}" + "".join(f"  {k:<{col_w}}" for k in REPORT_KEYS))
    print("-" * 75)
    for el, metric_stats in element_agg.items():
        row = f"  {el:<10}"
        for k in REPORT_KEYS:
            s = metric_stats.get(k, {})
            if s.get("mean") is not None:
                cell = f"{s['mean']:.3f}±{s['std']:.3f}(n={s['n']})"
            else:
                cell = "N/A"
            row += f"  {cell:<{col_w}}"
        print(row)
    print("=" * 75)


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def generate_plots_re_phys(records, out_dir):
    import warnings
    warnings.filterwarnings("ignore")

    metrics_labels = {
        "hungarian_rmsd":   "Hungarian RMSD (Å)",
        "rog_diff":         "RoG Difference (Å)",
        "rdf_wasserstein":  "RDF Wasserstein",
        "cn_wasserstein":   "CN Wasserstein",
    }

    # ── 1. Overall distribution (violin) ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Metric Distributions (all samples)", fontsize=13, fontweight="bold")
    for ax, (mk, label) in zip(axes, metrics_labels.items()):
        vals = [r[mk] for r in records if r.get(mk) is not None and np.isfinite(r[mk])]
        if vals:
            ax.violinplot(vals, showmedians=True)
            ax.set_title(label, fontsize=9)
            ax.set_xticks([])
        else:
            ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "re_phys_distributions.png", dpi=150)
    plt.close()

    # ── 2. Metric vs n_atoms (scatter + moving mean) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Metrics vs Cluster Size (all samples)", fontsize=13, fontweight="bold")
    for ax, (mk, label) in zip(axes.flat, metrics_labels.items()):
        xs = [r["n_atoms"] for r in records if r.get(mk) is not None and np.isfinite(r[mk])]
        ys = [r[mk]        for r in records if r.get(mk) is not None and np.isfinite(r[mk])]
        if not xs:
            continue
        xs, ys = np.array(xs), np.array(ys)
        ax.scatter(xs, ys, alpha=0.15, s=8, color="steelblue", rasterized=True)
        # bin means
        bins = np.arange(xs.min(), xs.max() + 2, 5)
        bin_means, bin_edges, _ = [], [], []
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
    plt.savefig(out_dir / "re_phys_vs_size.png", dpi=150)
    plt.close()

    # ── 3. Per-element bar chart (mean ± std) ──
    from collections import defaultdict
    elem_data = defaultdict(lambda: {mk: [] for mk in metrics_labels})
    for r in records:
        el = r.get("element", "?")
        for mk in metrics_labels:
            v = r.get(mk)
            if v is not None and np.isfinite(v):
                elem_data[el][mk].append(v)

    elements_sorted = sorted(elem_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(max(14, len(elements_sorted) * 0.35), 10))
    fig.suptitle("Per-Element Mean ± Std (all samples)", fontsize=13, fontweight="bold")
    for ax, (mk, label) in zip(axes.flat, metrics_labels.items()):
        means = [np.mean(elem_data[el][mk]) if elem_data[el][mk] else np.nan for el in elements_sorted]
        stds  = [np.std(elem_data[el][mk])  if elem_data[el][mk] else np.nan for el in elements_sorted]
        xs = np.arange(len(elements_sorted))
        ax.bar(xs, means, yerr=stds, capsize=2, color="steelblue", alpha=0.75, error_kw={"elinewidth": 0.8})
        ax.set_xticks(xs)
        ax.set_xticklabels(elements_sorted, rotation=90, fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "re_phys_by_element.png", dpi=150)
    plt.close()

    print(f"Saved plots -> {out_dir}/re_phys_*.png")


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

    # Parse targets
    tpat = re.compile(r"Target(?:\s+structure)?\s+(\d+)", re.IGNORECASE)
    epat = re.compile(r"element=(\w+)", re.IGNORECASE)

    targets_by_idx = {}
    for t in targets:
        m = tpat.search(t["comment"])
        if not m:
            continue
        tidx = int(m.group(1))
        em = epat.search(t["comment"])
        element = to_symbol(em.group(1)) if em else (t["atoms"][0] if t["atoms"] else "?")
        t["element"] = element
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
            n_skip += 1
            continue
        tidx = int(gm.group(1))
        sidx = int(sm.group(1)) if sm else -1

        if tidx not in targets_by_idx:
            n_skip += 1
            continue
        tgt = targets_by_idx[tidx]
        if gen["n_atoms"] != tgt["n_atoms"]:
            n_skip += 1
            continue

        m = compute_metrics(gen, tgt)
        m["target_idx"] = tidx
        m["sample_idx"] = sidx
        m["element"]    = tgt["element"]
        m["n_atoms"]    = tgt["n_atoms"]

        if tidx not in results_by_target:
            results_by_target[tidx] = {
                "element":            tgt["element"],
                "n_atoms":            tgt["n_atoms"],
                "target_comment":     tgt["comment"],
                "metrics_per_sample": [],
            }
        results_by_target[tidx]["metrics_per_sample"].append(m)

    n_mapped_tgts = len(results_by_target)
    n_mapped_gen  = sum(len(v["metrics_per_sample"]) for v in results_by_target.values())
    print(f"\n  Mapped {n_mapped_tgts} targets, {n_mapped_gen} samples  ({n_skip} skipped)")

    # Build flat records (all samples)
    all_sample_records = []
    for tidx, data in results_by_target.items():
        all_sample_records.extend(data["metrics_per_sample"])

    # Aggregate over ALL samples
    overall_agg  = aggregate(all_sample_records)
    size_bin_agg = aggregate(all_sample_records, key_fn=lambda r: size_bin(r["n_atoms"]))
    element_agg  = aggregate(all_sample_records, key_fn=lambda r: r["element"])

    # Terminal output
    print_overall(overall_agg)
    print_size_table(size_bin_agg)
    print_element_table(element_agg)

    # Save: raw_samples.csv
    csv_path = out_dir / "raw_samples.csv"
    csv_fields = [
        "target_idx", "sample_idx", "element", "n_atoms",
        "hungarian_rmsd", "rog_gen", "rog_tgt", "rog_diff",
        "rdf_wasserstein", "cn_wasserstein", "cn_cutoff_gen", "cn_cutoff_tgt",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_sample_records)
    print(f"\nSaved raw_samples.csv      -> {csv_path}  ({len(all_sample_records)} rows)")

    # Save: evaluation_detailed.json
    detailed = {}
    for tidx, data in results_by_target.items():
        samps = data["metrics_per_sample"]
        h_vals = [(s["hungarian_rmsd"] if s["hungarian_rmsd"] is not None else np.inf)
                  for s in samps]
        best_idx = int(np.argmin(h_vals)) if not all(np.isinf(h_vals)) else -1
        detailed[str(tidx)] = {
            "element":            data["element"],
            "n_atoms":            data["n_atoms"],
            "target_comment":     data["target_comment"],
            "best_sample_idx":    best_idx,
            "metrics_per_sample": samps,
        }
    det_path = out_dir / "evaluation_detailed.json"
    with open(det_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"Saved evaluation_detailed.json -> {det_path}")

    # Save: summary_overall.json
    summary = {
        "n_targets":   n_mapped_tgts,
        "n_generated": n_mapped_gen,
        "n_skipped":   n_skip,
        "overall":     overall_agg,
        "by_size_bin": size_bin_agg,
        "by_element":  element_agg,
    }
    sum_path = out_dir / "summary_overall.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary_overall.json     -> {sum_path}")


    # ── Plots ──
    generate_plots_re_phys(all_sample_records, out_dir)

    return summary


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="evaluate_re_phys: nanocluster reconstruction + physical metrics"
    )
    parser.add_argument("--generated", required=True, help="Generated structures XYZ")
    parser.add_argument("--target",    required=True, help="Target structures XYZ")
    parser.add_argument("--output",    default=None,  help="Output directory (default: same dir as generated)")
    args = parser.parse_args()
    evaluate(args.generated, args.target, args.output)


if __name__ == "__main__":
    main()
