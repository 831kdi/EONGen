"""
evaluate_re_lat.py
Latent-space similarity metrics for nanocluster generation evaluation.

Metrics:
  - SOAP Wasserstein distance:
      Per-atom SOAP descriptors computed with element-adaptive cutoff
      (r_cut = 4 * r_cov, sigma = 0.4 * r_cov, n_max=8, l_max=6).
      Structure-level similarity via Sliced Wasserstein distance between
      the distributions of per-atom SOAP vectors.

  - MACE cosine distance (optional, requires mace-torch):
      Latent embeddings from pretrained MACE-MP-0 (final interaction block).
      Structure-level: mean pooling -> cosine distance.

PCA visualisation:
  - Target and generated structures projected into 2D PCA space.
  - Plot 1: coloured by size bin  (01-10, 11-20, ..., 51-55)
  - Plot 2: coloured by element correlation cluster
             (7 groups derived from hierarchical clustering of the
              user-supplied correlation matrix)

Outputs:
  - lat_raw_samples.csv          : per-sample soap_wasserstein, mace_cosine_dist
  - lat_evaluation_detailed.json : per-target grouped results
  - lat_summary.json             : overall + size-bin + per-element stats
  - lat_pca_size.png             : PCA coloured by size bin
  - lat_pca_element_cluster.png  : PCA coloured by correlation cluster
  Terminal: overall + size-bin table
"""

import re
import os
import json
import csv
import argparse
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from dscribe.descriptors import SOAP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Constants & colour maps
# ─────────────────────────────────────────────────────────────

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

# 7 correlation-derived element clusters (from hierarchical clustering at k=7)
# Computed at startup from the user-supplied correlation CSV; fallback hardcoded.
CLUSTER_COLORS = {
    1: "#e41a1c",   # pnictogen  Bi/Sb/As/P
    2: "#ff7f00",   # chalcogen  Te/C/Se/S/Mo
    3: "#000000",   # outlier    B
    4: "#4daf4a",   # early TM   Fe/Zr/Hf/Y/Ca/Sr/Ba/Li/Mn/Sc/Ti
    5: "#377eb8",   # transition Na/K/Co/.../Ru
    6: "#984ea3",   # V/Cr
    7: "#a65628",   # refractory Ta/Nb/Re/W
}

CLUSTER_LABELS = {
    1: "Pnictogen (Bi/Sb/As/P)",
    2: "Chalcogen+C (Te/C/Se/S/Mo)",
    3: "B",
    4: "Early TM+Alk.Earth (Fe/Zr/...)",
    5: "Transition+sp metals",
    6: "V/Cr",
    7: "Refractory (Ta/Nb/Re/W)",
}

REPORT_KEYS_SOAP = ["soap_wasserstein"]
REPORT_KEYS_MACE = ["mace_cosine_dist"]


# ─────────────────────────────────────────────────────────────
# Element helpers
# ─────────────────────────────────────────────────────────────

def to_symbol(val):
    try:
        z = int(val)
        return chemical_symbols[z]
    except (ValueError, TypeError):
        return str(val)


def cov_radius(symbol):
    try:
        return float(covalent_radii[atomic_numbers[symbol]])
    except Exception:
        return 1.50


# ─────────────────────────────────────────────────────────────
# Build element -> correlation cluster mapping
# ─────────────────────────────────────────────────────────────

def build_element_clusters(corr_csv_path, k=7):
    """
    Returns dict {element_symbol: cluster_id (1-based)}.
    Falls back to empty dict if CSV not provided / unreadable.
    """
    if corr_csv_path is None or not Path(corr_csv_path).exists():
        return {}
    try:
        df = pd.read_csv(corr_csv_path, index_col=0)
        dist = np.clip(1.0 - df.values, 0, None)
        np.fill_diagonal(dist, 0)
        Z = linkage(squareform(dist), method='average')
        labels = fcluster(Z, k, criterion='maxclust')
        return {el: int(lbl) for el, lbl in zip(df.index, labels)}
    except Exception as e:
        print(f"  [warn] Could not build element clusters: {e}")
        return {}


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
# SOAP descriptor
# ─────────────────────────────────────────────────────────────

_soap_cache = {}   # (element, species_tuple) -> SOAP object

def get_soap(element_symbol, all_species):
    """
    Build (or retrieve cached) SOAP descriptor object for a given element.
    Parameters follow paper: r_cut = 4*r_cov, sigma = 0.4*r_cov, n_max=8, l_max=6.
    Since our clusters are single-element, species list is always [element_symbol].
    """
    species_key = tuple(sorted(all_species))
    cache_key = (element_symbol, species_key)
    if cache_key in _soap_cache:
        return _soap_cache[cache_key]

    r_cov   = cov_radius(element_symbol)
    r_cut   = 4.0 * r_cov
    sigma   = 0.4 * r_cov

    soap = SOAP(
        species=list(species_key),
        r_cut=r_cut,
        n_max=8,
        l_max=6,
        sigma=sigma,
        periodic=False,
        sparse=False,
    )
    _soap_cache[cache_key] = soap
    return soap


def compute_soap_vectors(atoms_list, positions, element_symbol):
    """
    Returns per-atom SOAP matrix of shape (N, D).
    """
    try:
        species = list(set(atoms_list))
        soap = get_soap(element_symbol, species)
        ase_atoms = Atoms(symbols=atoms_list, positions=positions)
        desc = soap.create(ase_atoms)          # (N, D)
        if desc.ndim == 1:
            desc = desc[np.newaxis, :]
        return desc.astype(np.float32)
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────
# Sliced Wasserstein distance between two SOAP distributions
# ─────────────────────────────────────────────────────────────

def sliced_wasserstein(X, Y, n_projections=64, seed=42):
    """
    Sliced Wasserstein distance between two point clouds X (N,D) and Y (M,D).
    Projects onto random unit vectors and averages 1D Wasserstein distances.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    directions = rng.standard_normal((n_projections, d)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms

    dists = []
    for v in directions:
        px = X @ v
        py = Y @ v
        dists.append(wasserstein_distance(px, py))
    return float(np.mean(dists))


def soap_wasserstein(gen, tgt, element_symbol):
    """
    Compute SOAP sliced Wasserstein distance between generated and target.
    Returns (distance, soap_gen_matrix, soap_tgt_matrix).
    """
    gen_pos = np.array(gen["positions"])
    tgt_pos = np.array(tgt["positions"])

    if not (np.isfinite(gen_pos).all() and np.abs(gen_pos).max() < 1000):
        return np.nan, None, None

    X = compute_soap_vectors(gen["atoms"], gen_pos, element_symbol)
    Y = compute_soap_vectors(tgt["atoms"], tgt_pos, element_symbol)

    if X is None or Y is None:
        return np.nan, X, Y

    try:
        d = sliced_wasserstein(X, Y)
        return (d if np.isfinite(d) else np.nan), X, Y
    except Exception:
        return np.nan, X, Y


# ─────────────────────────────────────────────────────────────
# MACE latent embeddings (optional)
# ─────────────────────────────────────────────────────────────

_mace_calc = None
_mace_available = False

def init_mace(model_size="small", device="cpu"):
    global _mace_calc, _mace_available
    try:
        from mace.calculators import mace_mp
        _mace_calc = mace_mp(
            model=model_size,
            device=device,
            default_dtype="float32",
        )
        _mace_available = True
        print(f"  MACE-MP-0 ({model_size}) loaded on {device}")
    except Exception as e:
        print(f"  [warn] MACE not available: {e}")
        print("         Skipping MACE metrics. Install with: pip install mace-torch")
        _mace_available = False


def get_mace_embedding(atoms_list, positions):
    """
    Extract mean-pooled latent embedding from MACE-MP-0.
    Returns 1D numpy array or None.
    """
    if not _mace_available or _mace_calc is None:
        return None
    try:
        import torch
        ase_atoms = Atoms(symbols=atoms_list, positions=positions)
        ase_atoms.calc = _mace_calc

        # Get descriptors (node features from final interaction block)
        desc = _mace_calc.get_descriptors(ase_atoms, invariants_only=True)
        # desc shape: (N_atoms, D)
        h_struct = desc.mean(axis=0)   # mean pooling
        return h_struct.astype(np.float32)
    except Exception as e:
        return None


def cosine_distance(a, b):
    """1 - cosine_similarity. Returns float in [0, 2]."""
    if a is None or b is None:
        return np.nan
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────
# Statistics & aggregation
# ─────────────────────────────────────────────────────────────

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


def size_bin_label(n):
    for lo, hi, label in SIZE_BINS:
        if lo <= n <= hi:
            return label
    return "51+"


def aggregate(records, keys, key_fn=None):
    if key_fn is None:
        return {k: stats([r.get(k) for r in records]) for k in keys}
    groups = {}
    for r in records:
        g = key_fn(r)
        groups.setdefault(g, []).append(r)
    return {g: aggregate(recs, keys) for g, recs in sorted(groups.items(), key=lambda x: str(x[0]))}


# ─────────────────────────────────────────────────────────────
# Terminal printing
# ─────────────────────────────────────────────────────────────

def print_summary(overall, size_agg, report_keys):
    W = 80
    print("\n" + "=" * W)
    print("  OVERALL SUMMARY  (all generated samples)")
    print("=" * W)
    for k in report_keys:
        s = overall.get(k, {})
        if s.get("mean") is not None:
            print(f"  {k:<30}: {s['mean']:8.4f} ± {s['std']:7.4f}  "
                  f"(median {s['median']:8.4f},  n={s['n']})")
        else:
            print(f"  {k:<30}: N/A")
    print("=" * W)

    col_w = 26
    print("\n" + "=" * W)
    print("  BY SIZE BIN")
    print("=" * W)
    print(f"  {'Bin':<10}" + "".join(f"  {k:<{col_w}}" for k in report_keys))
    print("-" * W)
    for bin_label, mstats in size_agg.items():
        row = f"  {bin_label:<10}"
        for k in report_keys:
            s = mstats.get(k, {})
            if s.get("mean") is not None:
                cell = f"{s['mean']:.3f}±{s['std']:.3f}(n={s['n']})"
            else:
                cell = "N/A"
            row += f"  {cell:<{col_w}}"
        print(row)
    print("=" * W)


# ─────────────────────────────────────────────────────────────
# PCA Plots
# ─────────────────────────────────────────────────────────────

def make_pca_plots(pca_records, out_dir):
    """
    pca_records: list of dicts with keys:
        embedding (np.array 1D), n_atoms, element, el_cluster,
        is_target (bool), target_idx, sample_idx
    """
    if not pca_records:
        print("  [warn] No PCA records to plot.")
        return

    # Stack all embeddings and fit PCA on ALL data (targets + generated)
    embeddings = np.stack([r["embedding"] for r in pca_records])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)   # (N, 2)
    var = pca.explained_variance_ratio_

    targets_mask   = np.array([r["is_target"] for r in pca_records])
    generated_mask = ~targets_mask

    # ── Plot 1: coloured by size bin ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        f"SOAP PCA — coloured by size bin\n"
        f"(PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%)",
        fontsize=12, fontweight="bold"
    )

    for lo, hi, label in SIZE_BINS:
        color = SIZE_COLORS[label]
        # generated
        mask = generated_mask & np.array([lo <= r["n_atoms"] <= hi for r in pca_records])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, alpha=0.35, s=18, marker="o",
                       label=f"gen {label}", rasterized=True, linewidths=0)
        # targets
        mask_t = targets_mask & np.array([lo <= r["n_atoms"] <= hi for r in pca_records])
        if mask_t.any():
            ax.scatter(coords[mask_t, 0], coords[mask_t, 1],
                       c=color, alpha=1.0, s=120, marker="*",
                       edgecolors="black", linewidths=0.5)

    # legend: size bins (generated = circle, targets = star)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SIZE_COLORS[lbl],
               markersize=8, label=f"gen {lbl}", alpha=0.7)
        for _, _, lbl in SIZE_BINS
    ] + [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=12, label="target ★")
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8, ncol=2)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path1 = out_dir / "lat_pca_size.png"
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  Saved {path1}")

    # ── Plot 2: coloured by element correlation cluster ───────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        f"SOAP PCA — coloured by element correlation cluster\n"
        f"(PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%)",
        fontsize=12, fontweight="bold"
    )

    all_clusters = sorted(set(r["el_cluster"] for r in pca_records if r["el_cluster"] is not None))

    for cl in all_clusters:
        color = CLUSTER_COLORS.get(cl, "#888888")
        label = CLUSTER_LABELS.get(cl, f"Cluster {cl}")
        # generated
        mask = generated_mask & np.array([r["el_cluster"] == cl for r in pca_records])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, alpha=0.35, s=18, marker="o",
                       label=f"gen: {label}", rasterized=True, linewidths=0)
        # targets
        mask_t = targets_mask & np.array([r["el_cluster"] == cl for r in pca_records])
        if mask_t.any():
            ax.scatter(coords[mask_t, 0], coords[mask_t, 1],
                       c=color, alpha=1.0, s=140, marker="*",
                       edgecolors="black", linewidths=0.5)

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.legend(loc="best", fontsize=6.5, ncol=1)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path2 = out_dir / "lat_pca_element_cluster.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved {path2}")


# ─────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────

def evaluate(
    generated_file,
    target_file,
    corr_csv=None,
    output_dir=None,
    use_mace=False,
    mace_device="cpu",
    mace_model="small",
):
    out_dir = Path(output_dir) if output_dir else Path(generated_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build element clusters from correlation CSV
    el_to_cluster = build_element_clusters(corr_csv)
    if el_to_cluster:
        print(f"  Element clusters loaded from {corr_csv} (k=7)")
    else:
        print("  No correlation CSV — element cluster colouring disabled")

    # Optionally load MACE
    if use_mace:
        init_mace(model_size=mace_model, device=mace_device)

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

    # Parse generated & evaluate
    gpat = re.compile(r"\bTarget\s+(\d+)\b", re.IGNORECASE)
    spat = re.compile(r"\bsample\s+(\d+)\b", re.IGNORECASE)

    results_by_target = {}
    n_skip = 0

    # For PCA: collect SOAP mean vectors for all structures
    pca_records = []

    # Pre-compute target SOAP and MACE embeddings (avoid re-computing per sample)
    target_soap  = {}   # tidx -> np.array (N_tgt, D)
    target_mace  = {}   # tidx -> np.array (D,)

    print("\nPre-computing target descriptors...")
    for tidx, tgt in tqdm(targets_by_idx.items(), desc="Targets"):
        el  = tgt["element"]
        pos = np.array(tgt["positions"])

        # SOAP
        Y = compute_soap_vectors(tgt["atoms"], pos, el)
        target_soap[tidx] = Y

        # MACE
        if use_mace:
            h = get_mace_embedding(tgt["atoms"], pos)
            target_mace[tidx] = h

        # PCA record for target (use mean SOAP vector as embedding)
        if Y is not None:
            pca_records.append({
                "embedding":   Y.mean(axis=0),
                "n_atoms":     tgt["n_atoms"],
                "element":     el,
                "el_cluster":  el_to_cluster.get(el),
                "is_target":   True,
                "target_idx":  tidx,
                "sample_idx":  -1,
            })

    print("\nEvaluating generated structures...")
    for gen in tqdm(generated, desc="Generated"):
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

        el  = tgt["element"]
        pos = np.array(gen["positions"])

        if not (np.isfinite(pos).all() and np.abs(pos).max() < 1000):
            n_skip += 1; continue

        # ── SOAP Wasserstein ──────────────────────────────
        X = compute_soap_vectors(gen["atoms"], pos, el)
        Y = target_soap.get(tidx)

        soap_w = np.nan
        if X is not None and Y is not None:
            try:
                soap_w = sliced_wasserstein(X, Y)
                if not np.isfinite(soap_w):
                    soap_w = np.nan
            except Exception:
                pass

        # ── MACE cosine distance ──────────────────────────
        mace_dist = np.nan
        mace_emb_gen = None
        if use_mace:
            mace_emb_gen = get_mace_embedding(gen["atoms"], pos)
            mace_emb_tgt = target_mace.get(tidx)
            mace_dist    = cosine_distance(mace_emb_gen, mace_emb_tgt)

        # ── Record ────────────────────────────────────────
        rec = {
            "target_idx":       tidx,
            "sample_idx":       sidx,
            "element":          el,
            "n_atoms":          tgt["n_atoms"],
            "soap_wasserstein": float(soap_w) if np.isfinite(soap_w) else None,
            "mace_cosine_dist": float(mace_dist) if np.isfinite(mace_dist) else None,
        }

        if tidx not in results_by_target:
            results_by_target[tidx] = {
                "element":            el,
                "n_atoms":            tgt["n_atoms"],
                "target_comment":     tgt["comment"],
                "metrics_per_sample": [],
            }
        results_by_target[tidx]["metrics_per_sample"].append(rec)

        # PCA record for generated (use mean SOAP as embedding)
        if X is not None:
            pca_records.append({
                "embedding":   X.mean(axis=0),
                "n_atoms":     gen["n_atoms"],
                "element":     el,
                "el_cluster":  el_to_cluster.get(el),
                "is_target":   False,
                "target_idx":  tidx,
                "sample_idx":  sidx,
            })

    n_mapped_tgts = len(results_by_target)
    n_mapped_gen  = sum(len(v["metrics_per_sample"]) for v in results_by_target.values())
    print(f"\n  Mapped {n_mapped_tgts} targets, {n_mapped_gen} samples  ({n_skip} skipped)")

    # Flat records
    all_records = []
    for data in results_by_target.values():
        all_records.extend(data["metrics_per_sample"])

    # Determine which metric keys are actually populated
    active_keys = []
    for k in ["soap_wasserstein", "mace_cosine_dist"]:
        if any(r.get(k) is not None for r in all_records):
            active_keys.append(k)

    # Aggregate
    overall_agg  = aggregate(all_records, active_keys)
    size_bin_agg = aggregate(all_records, active_keys, key_fn=lambda r: size_bin_label(r["n_atoms"]))
    element_agg  = aggregate(all_records, active_keys, key_fn=lambda r: r["element"])

    # Terminal
    print_summary(overall_agg, size_bin_agg, active_keys)

    # ── PCA plots ─────────────────────────────────────────
    print("\nGenerating PCA plots...")
    make_pca_plots(pca_records, out_dir)

    # ── Save: lat_raw_samples.csv ──────────────────────────
    csv_path = out_dir / "lat_raw_samples.csv"
    csv_fields = ["target_idx", "sample_idx", "element", "n_atoms",
                  "soap_wasserstein", "mace_cosine_dist"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Saved lat_raw_samples.csv          -> {csv_path}  ({len(all_records)} rows)")

    # ── Save: lat_evaluation_detailed.json ────────────────
    detailed = {}
    for tidx, data in results_by_target.items():
        samps = data["metrics_per_sample"]
        sw_vals = [(s["soap_wasserstein"] if s["soap_wasserstein"] is not None else np.inf)
                   for s in samps]
        best_idx = int(np.argmin(sw_vals)) if not all(np.isinf(sw_vals)) else -1
        detailed[str(tidx)] = {
            "element":            data["element"],
            "n_atoms":            data["n_atoms"],
            "target_comment":     data["target_comment"],
            "best_sample_idx":    best_idx,
            "metrics_per_sample": samps,
        }
    det_path = out_dir / "lat_evaluation_detailed.json"
    with open(det_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"Saved lat_evaluation_detailed.json -> {det_path}")

    # ── Save: lat_summary.json ─────────────────────────────
    summary = {
        "n_targets":   n_mapped_tgts,
        "n_generated": n_mapped_gen,
        "n_skipped":   n_skip,
        "mace_used":   use_mace and _mace_available,
        "overall":     overall_agg,
        "by_size_bin": size_bin_agg,
        "by_element":  element_agg,
    }
    sum_path = out_dir / "lat_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved lat_summary.json             -> {sum_path}")

    return summary


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="evaluate_re_lat: latent-space similarity metrics for nanoclusters"
    )
    parser.add_argument("--generated",    required=True,  help="Generated structures XYZ")
    parser.add_argument("--target",       required=True,  help="Target structures XYZ")
    parser.add_argument("--corr_csv",     default=None,   help="Element correlation CSV (for PCA colouring)")
    parser.add_argument("--output",       default=None,   help="Output directory")
    parser.add_argument("--use_mace",     action="store_true", help="Compute MACE cosine distances")
    parser.add_argument("--mace_device",  default="cpu",  help="Device for MACE (cpu / cuda)")
    parser.add_argument("--mace_model",   default="small", choices=["small", "medium", "large"],
                        help="MACE-MP-0 model size")
    args = parser.parse_args()

    evaluate(
        generated_file = args.generated,
        target_file    = args.target,
        corr_csv       = args.corr_csv,
        output_dir     = args.output,
        use_mace       = args.use_mace,
        mace_device    = args.mace_device,
        mace_model     = args.mace_model,
    )


if __name__ == "__main__":
    main()
