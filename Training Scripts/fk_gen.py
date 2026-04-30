"""
Steered generation for EONGen (v7 compatible)
=====================================================================
Wraps Euler ODE integration with SMC reweighting to steer generated
nanoclusters toward a target coordination-number regime.

Steering modes
--------------
  high_cn  : bias toward compact / bulk-like structures (high mean CN)
  low_cn   : bias toward open / under-coordinated structures (low mean CN)
  none     : plain CFM generation (FK disabled)

Note on GoTenNet batching
--------------------------
GoTenNet does not support multi-graph batching in this setup, so
particles are evaluated one at a time. Use --n_particles 8 and
--max_targets 20 for a run that completes in ~2h.

Usage examples
--------------
  # Baseline (no steering)
  python fk_gen.py \\
      --checkpoint checkpoints_7/best/best_model.pt \\
      --test_data  processed_data_n20/test_dataset.pkl \\
      --output_dir results_fk/none \\
      --mode none --n_particles 8 --n_samples 10 --max_targets 20

  # High CN steering
  python fk_gen.py \\
      --checkpoint checkpoints_7/best/best_model.pt \\
      --test_data  processed_data_n20/test_dataset.pkl \\
      --output_dir results_fk/high_cn \\
      --mode high_cn --gamma 1.0 --n_particles 8 --n_samples 10 --max_targets 20

  # Low CN steering
  python fk_gen.py \\
      --checkpoint checkpoints_7/best/best_model.pt \\
      --test_data  processed_data_n20/test_dataset.pkl \\
      --output_dir results_fk/low_cn \\
      --mode low_cn --gamma 1.0 --n_particles 8 --n_samples 10 --max_targets 20
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from torch_geometric.data import Data
from tqdm import tqdm

# ── import model ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from v7_train import NanoparticleCFM, NanoparticleDataset
except ImportError:
    raise ImportError(
        "Cannot import from v7_train.py. Make sure it is in the same directory."
    )

ELEM_TO_Z = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
    "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,
    "Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
    "Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,
    "Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,
    "Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,
    "Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,"W":74,
    "Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,
    "Bi":83,"Po":84,"At":85,"Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,
    "Pa":91,"U":92,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Potential functions  (all on CPU, geometry only)
# ═══════════════════════════════════════════════════════════════════════════════

from ase.data import covalent_radii as _cov_radii


def _pairwise_cutoff_matrix(z_val: int, n_atoms: int, scale: float = 1.2) -> torch.Tensor:
    """
    Returns (N, N) tensor of pairwise cutoffs = scale * (r_cov_i + r_cov_j).
    For a homoatomic cluster all entries are identical.
    """
    r = _cov_radii[z_val]
    cutoff = scale * 2.0 * r
    return torch.full((n_atoms, n_atoms), cutoff, dtype=torch.float32)


def compute_cn(pos: torch.Tensor, z_val: int, scale: float = 1.2) -> float:
    """Mean coordination number using element-specific covalent radii cutoff."""
    pos = pos.detach().float()
    n = pos.shape[0]
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N,N,3)
    dist = diff.norm(dim=-1)                       # (N,N)
    cutoff_mat = _pairwise_cutoff_matrix(z_val, n, scale).to(pos.device)
    mask = (dist < cutoff_mat) & (dist > 1e-6)
    return mask.float().sum(dim=-1).mean().item()


def connectivity_penalty(pos: torch.Tensor, z_val: int, scale: float = 1.2) -> float:
    """Sum of ReLU(1 - CN_i): penalises isolated atoms."""
    pos = pos.detach().float()
    n = pos.shape[0]
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist = diff.norm(dim=-1)
    cutoff_mat = _pairwise_cutoff_matrix(z_val, n, scale).to(pos.device)
    mask = (dist < cutoff_mat) & (dist > 1e-6)
    cn_per_atom = mask.float().sum(dim=-1)
    return torch.relu(1.0 - cn_per_atom).sum().item()


def potential(pos: torch.Tensor, mode: str, z_val: int,
              lambda_conn: float, scale: float = 1.2) -> float:
    """
    Returns V(x).  FK uses w ∝ exp(-gamma * V), so lower V = higher weight.
      high_cn : V = -mean_CN          (high CN → low V → high weight)
      low_cn  : V =  mean_CN + lambda_conn * connectivity_penalty
      none    : V = 0
    """
    if mode == "none":
        return 0.0
    cn = compute_cn(pos, z_val, scale)
    if mode == "high_cn":
        return -cn
    else:  # low_cn
        pen = connectivity_penalty(pos, z_val, scale)
        return cn + lambda_conn * pen


# ═══════════════════════════════════════════════════════════════════════════════
# SMC resampling
# ═══════════════════════════════════════════════════════════════════════════════

def systematic_resample(particles: list, log_weights: np.ndarray):
    """
    Systematic resampling.
    particles : list of (N,3) tensors
    log_weights: (P,) unnormalised
    Returns (resampled list, ESS float).
    """
    P = len(log_weights)
    log_w = log_weights - logsumexp(log_weights)
    w = np.exp(log_w)
    w = w / w.sum()
    ess = 1.0 / (w ** 2).sum()

    positions = (np.arange(P) + np.random.uniform()) / P
    cumsum = np.cumsum(w)
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, P - 1)

    return [particles[i].clone() for i in indices], ess


# ═══════════════════════════════════════════════════════════════════════════════
# Single-particle velocity query  (GoTenNet only supports batch_size=1)
# ═══════════════════════════════════════════════════════════════════════════════

def get_velocity(model, pos: torch.Tensor, z: torch.Tensor,
                 t_val: float, device) -> torch.Tensor:
    """pos: (N,3) on device, z: (N,) on device → velocity (N,3) on device."""
    n = pos.shape[0]
    data = Data(
        pos=pos,
        z=z,
        batch=torch.zeros(n, dtype=torch.long, device=device),
    )
    t = torch.tensor([t_val], device=device)
    with torch.no_grad():
        return model(data, t)  # (N,3)


# ═══════════════════════════════════════════════════════════════════════════════
# FK-steered generation for a single target
# ═══════════════════════════════════════════════════════════════════════════════

def fk_generate_one(model, z: torch.Tensor, n_atoms: int, device,
                    n_particles: int, n_steps: int,
                    mode: str, gamma: float, z_val: int,
                    lambda_conn: float, resample_every: int) -> torch.Tensor:
    """
    Returns best structure (N,3) cpu tensor.
    Each particle is evaluated separately to work around GoTenNet batch limit.
    """
    # Initialise P particles
    particles = [torch.randn(n_atoms, 3, device=device) for _ in range(n_particles)]

    dt = 1.0 / n_steps
    ts = np.linspace(0.0, 1.0, n_steps + 1)

    for step in range(n_steps):
        t_val = float(ts[step])

        # Euler step for each particle
        new_particles = []
        for pos in particles:
            vel = get_velocity(model, pos, z, t_val, device)
            pos_new = pos + dt * vel
            # Centre (remove COM drift)
            pos_new = pos_new - pos_new.mean(dim=0, keepdim=True)
            new_particles.append(pos_new)
        particles = new_particles

        # FK reweighting
        if mode != "none" and (step + 1) % resample_every == 0:
            log_w = np.array([
                -gamma * potential(p.cpu(), mode, z_val, lambda_conn)
                for p in particles
            ])
            particles, ess = systematic_resample(particles, log_w)

    # Pick best particle (lowest potential)
    if mode == "none":
        best = particles[0]
    else:
        V = [potential(p.cpu(), mode, z_val, lambda_conn) for p in particles]
        best = particles[int(np.argmin(V))]

    return best.cpu()


# ═══════════════════════════════════════════════════════════════════════════════
# XYZ save
# ═══════════════════════════════════════════════════════════════════════════════

def save_xyz(pos: np.ndarray, element: str, filepath: Path, comment: str = ""):
    lines = [str(pos.shape[0]), comment]
    for coord in pos:
        lines.append(f"{element}  {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f}")
    filepath.write_text("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt       = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    model = NanoparticleCFM(
        hidden_dim     = saved_args.get("hidden_dim",     256),
        n_interactions = saved_args.get("n_interactions", 4),
        condition_dim  = saved_args.get("condition_dim",  128),
        cutoff         = saved_args.get("cutoff",         5.0),
        max_neighbors  = saved_args.get("max_neighbors",  128),
        use_gotennet   = saved_args.get("use_gotennet",   True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss',0):.4f}")

    # Build target list
    targets = []
    if args.test_data:
        with open(args.test_data, "rb") as f:
            dataset = pickle.load(f)
        seen = set()
        for sample in dataset:
            elem   = sample.element if hasattr(sample, "element") else sample.get("element")
            natoms = sample.n_atoms if hasattr(sample, "n_atoms") else sample.get("n_atoms")
            key = (elem, natoms)
            if key not in seen:
                seen.add(key)
                targets.append(key)
        print(f"Loaded {len(targets)} unique (element, n_atoms) targets.")

        if args.max_targets and args.max_targets < len(targets):
            step = len(targets) / args.max_targets
            targets = [targets[int(i * step)] for i in range(args.max_targets)]
            print(f"Subsampled to {len(targets)} targets (--max_targets {args.max_targets}).")

    elif args.element and args.n_atoms:
        targets = [(args.element, args.n_atoms)]
    else:
        raise ValueError("Provide --test_data or both --element and --n_atoms.")

    # Estimate runtime
    sec_per_step = 0.637 / 8  # measured: 8 particles/step ≈ 0.637s
    total_steps  = len(targets) * args.n_samples * args.n_steps * args.n_particles
    est_h        = total_steps * sec_per_step / 3600
    print(f"Estimated runtime: {est_h:.1f}h  "
          f"({len(targets)} targets × {args.n_samples} samples × "
          f"{args.n_steps} steps × {args.n_particles} particles)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cn = []
    all_records = []

    for elem, n_atoms in tqdm(targets, desc="Targets"):
        z_val = ELEM_TO_Z.get(elem)
        if z_val is None:
            print(f"[WARN] Unknown element {elem}, skipping.")
            continue
        z = torch.full((n_atoms,), z_val, dtype=torch.long, device=device)

        target_dir = out_dir / f"{elem}_{n_atoms}"
        target_dir.mkdir(exist_ok=True)

        for s in range(args.n_samples):
            pos = fk_generate_one(
                model=model, z=z, n_atoms=n_atoms, device=device,
                n_particles=args.n_particles, n_steps=args.n_steps,
                mode=args.mode, gamma=args.gamma,
                z_val=z_val, lambda_conn=args.lambda_conn,
                resample_every=args.resample_every,
            )  # (N,3) cpu

            cn_val = compute_cn(pos, z_val)
            all_cn.append(cn_val)
            all_records.append({"element": elem, "n_atoms": n_atoms,
                                 "sample": s, "mean_cn": cn_val})

            save_xyz(
                pos.numpy(), elem,
                target_dir / f"sample_{s:04d}.xyz",
                comment=f"mode={args.mode} gamma={args.gamma} cn={cn_val:.3f}",
            )

    # Summary
    print(f"\nDone. {len(all_cn)} structures → {out_dir}")
    if all_cn:
        print(f"  CN  mean={np.mean(all_cn):.3f}  std={np.std(all_cn):.3f}"
              f"  min={np.min(all_cn):.3f}  max={np.max(all_cn):.3f}")

    # Ensure all values are JSON-serializable
    clean_records = [
        {k: float(v) if isinstance(v, (torch.Tensor, np.floating, np.integer)) else v
         for k, v in r.items()}
        for r in all_records
    ]
    summary = {
        "mode":        args.mode,
        "gamma":       float(args.gamma),
        "cn_cutoff":   "covalent_radii_based (scale=1.2)",
        "n_particles": int(args.n_particles),
        "n_steps":     int(args.n_steps),
        "n_samples":   int(args.n_samples),
        "max_targets": args.max_targets,
        "n_generated": int(len(all_cn)),
        "mean_cn": float(np.mean(all_cn)) if all_cn else None,
        "std_cn":  float(np.std(all_cn))  if all_cn else None,
        "min_cn":  float(np.min(all_cn))  if all_cn else None,
        "max_cn":  float(np.max(all_cn))  if all_cn else None,
        "records": clean_records,
    }
    with open(out_dir / "fk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {out_dir}/fk_summary.json")

    # CN distribution plot
    if all_cn:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=400)
        ax.hist(all_cn, bins=20, color="#5FABA2", edgecolor="white", linewidth=0.4)
        ax.axvline(float(np.mean(all_cn)), color="#D4447E", linewidth=1.2,
                   linestyle="--", label=f"mean={np.mean(all_cn):.2f}")
        ax.set_xlabel("Mean CN per structure", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(f"CN distribution  [mode={args.mode}]", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        plot_path = out_dir / "cn_distribution.png"
        fig.savefig(plot_path, dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"CN plot -> {plot_path}")


def main():
    p = argparse.ArgumentParser(description="FK-steered EONGen generation")

    # Model / data
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--test_data",    default=None)
    p.add_argument("--element",      default=None)
    p.add_argument("--n_atoms",      type=int, default=None)
    p.add_argument("--output_dir",   required=True)

    # Generation
    p.add_argument("--n_samples",    type=int,   default=10)
    p.add_argument("--n_steps",      type=int,   default=100)
    p.add_argument("--max_targets",  type=int,   default=None,
                   help="Subsample this many targets from test set (evenly spaced)")

    # FK steering
    p.add_argument("--mode",          default="low_cn",
                   choices=["high_cn", "low_cn", "none"])
    p.add_argument("--n_particles",   type=int,   default=8,
                   help="SMC particle count per sample")
    p.add_argument("--gamma",         type=float, default=1.0,
                   help="Steering strength")
    p.add_argument("--lambda_conn",   type=float, default=5.0,
                   help="Connectivity penalty weight (low_cn only)")
    p.add_argument("--resample_every",type=int,   default=10,
                   help="Resample every N ODE steps")

    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
