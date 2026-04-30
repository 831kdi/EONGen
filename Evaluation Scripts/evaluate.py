"""
Nanoparticle structures against ground truth.
"""
import re
import torch
import numpy as np
from pathlib import Path
import argparse
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import json
from tqdm import tqdm
from ase import Atoms
from ase.neighborlist import NeighborList


def read_xyz_file(filename):
    """Robust XYZ reader that tolerates blank lines and partial/truncated blocks."""
    structures = []
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    n_lines = len(lines)

    def _is_int_line(s):
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
        if not _is_int_line(lines[i]):
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        if n_atoms <= 0:
            i += 1
            continue
        if i + 1 >= n_lines:
            break
        comment = lines[i + 1].strip()
        start = i + 2
        end = start + n_atoms
        if end > n_lines:
            break
        atoms, positions, ok = [], [], True
        for j in range(start, end):
            parts = lines[j].split()
            if len(parts) < 4:
                ok = False
                break
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                ok = False
                break
            atoms.append(parts[0])
            positions.append([x, y, z])
        if ok:
            structures.append({
                "atoms": atoms,
                "positions": np.array(positions, dtype=float),
                "comment": comment,
                "n_atoms": n_atoms,
            })
            i = end
        else:
            i += 1
    return structures


def align_structures(pos1, pos2):
    """
    Kabsch alignment. Returns (aligned_pos1, rmsd).
    Returns (pos1, nan) if input is invalid or SVD fails.
    """
    # Guard: nan/inf → return nan immediately
    if not (np.isfinite(pos1).all() and np.isfinite(pos2).all()):
        return pos1, np.nan

    p = pos1 - pos1.mean(axis=0)
    q = pos2 - pos2.mean(axis=0)
    H = p.T @ q
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return pos1, np.nan

    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    p_rot = p @ R.T
    rmsd = float(np.sqrt(np.mean(np.sum((p_rot - q) ** 2, axis=1))))
    if not np.isfinite(rmsd):
        return pos1, np.nan
    return p_rot + pos2.mean(axis=0), rmsd


def compute_hungarian_rmsd(pos1, pos2):
    """Hungarian-matched Kabsch RMSD. Returns nan on failure."""
    if not (np.isfinite(pos1).all() and np.isfinite(pos2).all()):
        return np.nan
    try:
        dist_matrix = cdist(pos1, pos2)
        if not np.isfinite(dist_matrix).all():
            return np.nan
        row_ind, _ = linear_sum_assignment(dist_matrix)
        _, rmsd = align_structures(pos1[row_ind], pos2)
        return rmsd if np.isfinite(rmsd) else np.nan
    except Exception:
        return np.nan


def compute_chamfer_distance(pos1, pos2):
    """Chamfer distance. Returns nan on failure."""
    if not (np.isfinite(pos1).all() and np.isfinite(pos2).all()):
        return np.nan
    try:
        dist_matrix = cdist(pos1, pos2)
        if not np.isfinite(dist_matrix).all():
            return np.nan
        cd = dist_matrix.min(axis=1).mean() + dist_matrix.min(axis=0).mean()
        return float(cd) if np.isfinite(cd) else np.nan
    except Exception:
        return np.nan


def compute_structure_metrics(generated_pos, target_pos):
    """
    Returns dict of metrics. All values are float or None (never inf or large float).
    """
    # Guard: invalid generated structure (nan/inf or exploding coords) → all None
    if not np.isfinite(generated_pos).all() or np.abs(generated_pos).max() > 1000:
        return {k: None for k in
                ['rmsd', 'hungarian_rmsd', 'chamfer_distance',
                 'radius_of_gyration_diff', 'rog_generated', 'rog_target']}

    _, rmsd          = align_structures(generated_pos, target_pos)
    hungarian_rmsd   = compute_hungarian_rmsd(generated_pos, target_pos)
    chamfer          = compute_chamfer_distance(generated_pos, target_pos)

    try:
        rog_gen    = float(np.sqrt(np.mean(np.sum((generated_pos - generated_pos.mean(0))**2, axis=1))))
        rog_target = float(np.sqrt(np.mean(np.sum((target_pos   - target_pos.mean(0))**2,    axis=1))))
        rog_diff   = abs(rog_gen - rog_target)
        if not np.isfinite([rog_gen, rog_target, rog_diff]).all():
            rog_gen = rog_target = rog_diff = np.nan
    except Exception:
        rog_gen = rog_target = rog_diff = np.nan

    def _v(x):
        return float(x) if (x is not None and np.isfinite(x)) else None

    return {
        'rmsd':                   _v(rmsd),
        'hungarian_rmsd':         _v(hungarian_rmsd),
        'chamfer_distance':       _v(chamfer),
        'radius_of_gyration_diff':_v(rog_diff),
        'rog_generated':          _v(rog_gen),
        'rog_target':             _v(rog_target),
    }


def evaluate_generation(generated_file, target_file, output_file=None):
    print(f"Loading generated structures from {generated_file}...")
    generated = read_xyz_file(generated_file)
    print(f"  {len(generated)} generated structures loaded")

    print(f"Loading target structures from {target_file}...")
    targets = read_xyz_file(target_file)
    print(f"  {len(targets)} target structures loaded")

    # Build targets_by_idx  (comment must contain "Target structure N" or "Target N")
    tpat = re.compile(r"Target(?:\s+structure)?\s+(\d+)", re.IGNORECASE)
    targets_by_idx = {}
    for t in targets:
        m = tpat.search(t.get("comment", ""))
        if m:
            targets_by_idx[int(m.group(1))] = t
    print(f"  {len(targets_by_idx)} targets with index tag  "
          f"({len(targets) - len(targets_by_idx)} missing tag)")

    # Match generated → target
    gpat = re.compile(r"\bTarget\s+(\d+)\b", re.IGNORECASE)
    results_by_target = {}
    n_no_match = 0

    for gen in tqdm(generated, desc="Matching generated -> target"):
        m = gpat.search(gen.get("comment", ""))
        if m is None:
            n_no_match += 1
            continue
        tidx = int(m.group(1))
        if tidx not in targets_by_idx:
            n_no_match += 1
            continue
        tgt = targets_by_idx[tidx]
        if gen["n_atoms"] != tgt["n_atoms"]:
            n_no_match += 1
            continue
        metrics = compute_structure_metrics(
            np.array(gen["positions"]),
            np.array(tgt["positions"]),
        )
        if tidx not in results_by_target:
            results_by_target[tidx] = {"target": tgt, "metrics_per_sample": []}
        results_by_target[tidx]["metrics_per_sample"].append(metrics)

    n_mapped_targets   = len(results_by_target)
    n_mapped_generated = sum(len(v["metrics_per_sample"]) for v in results_by_target.values())
    spt = n_mapped_generated / n_mapped_targets if n_mapped_targets > 0 else 0.0

    print(f"\n[INFO] Mapped {n_mapped_targets} targets, {n_mapped_generated} samples  "
          f"({n_no_match} unmatched)")
    print(f"  Samples per mapped target: {spt:.2f}")

    # Aggregate
    all_metrics  = {k: [] for k in ['rmsd', 'hungarian_rmsd', 'chamfer_distance', 'radius_of_gyration_diff']}
    best_metrics = {k: [] for k in ['rmsd', 'hungarian_rmsd', 'chamfer_distance', 'radius_of_gyration_diff']}

    for tidx, data in results_by_target.items():
        for m in data["metrics_per_sample"]:
            for key in all_metrics:
                all_metrics[key].append(m[key])

        h_vals = [(m["hungarian_rmsd"] if m["hungarian_rmsd"] is not None else np.inf)
                  for m in data["metrics_per_sample"]]
        if not all(np.isinf(h_vals)):
            best = data["metrics_per_sample"][int(np.argmin(h_vals))]
            for key in best_metrics:
                best_metrics[key].append(best[key])

    summary = {
        "n_targets_in_file":     len(targets),
        "n_generated_in_file":   len(generated),
        "n_mapped_targets":      n_mapped_targets,
        "n_mapped_generated":    n_mapped_generated,
        "samples_per_mapped_target": spt,
    }

    def _report(label, metrics_dict, prefix):
        print(f"\n--- {label} ---")
        for metric_name, values in metrics_dict.items():
            valid  = [v for v in values if v is not None and np.isfinite(v)]
            n_excl = len(values) - len(valid)
            excl_s = f"  [{n_excl} excluded]" if n_excl > 0 else ""
            summary[f"{prefix}_{metric_name}_n_excluded"] = n_excl
            if not valid:
                print(f"{metric_name:30s}: No valid values{excl_s}")
                summary[f"{prefix}_{metric_name}_n_valid"] = 0
                continue
            mean, std, median = np.mean(valid), np.std(valid), np.median(valid)
            summary[f"{prefix}_{metric_name}_mean"]    = float(mean)
            summary[f"{prefix}_{metric_name}_std"]     = float(std)
            summary[f"{prefix}_{metric_name}_median"]  = float(median)
            summary[f"{prefix}_{metric_name}_n_valid"] = len(valid)
            print(f"{metric_name:30s}: {mean:8.4f} ± {std:8.4f}  (median: {median:8.4f}){excl_s}")

    print("\nComputing statistics...")
    _report("All Generated Samples",  all_metrics,  "all")
    _report("Best Sample Per Target", best_metrics, "best")
    print("=" * 80)

    if output_file:
        output_file = Path(output_file)
        with open(output_file.parent / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary  → {output_file.parent / 'evaluation_summary.json'}")

        detailed = {}
        for tidx, data in results_by_target.items():
            h_vals = [(m["hungarian_rmsd"] if (m.get("hungarian_rmsd") is not None
                       and np.isfinite(m["hungarian_rmsd"])) else np.inf)
                      for m in data["metrics_per_sample"]]
            best_idx = int(np.argmin(h_vals)) if not all(np.isinf(h_vals)) else -1
            detailed[str(tidx)] = {
                "metrics_per_sample": data["metrics_per_sample"],
                "best_sample_idx": best_idx,
            }
        with open(output_file, "w") as f:
            json.dump(detailed, f, indent=2)
        print(f"Saved detailed → {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated nanoparticle structures")
    parser.add_argument("--generated", type=str, required=True)
    parser.add_argument("--target",    type=str, required=True)
    parser.add_argument("--output",    type=str, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(args.generated).parent / "evaluation_detailed.json"
    evaluate_generation(args.generated, args.target, args.output)


if __name__ == "__main__":
    main()
