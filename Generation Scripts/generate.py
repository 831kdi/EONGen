"""
n script for NanoparticleCFM.

- Reads targets from test_dataset.pkl
- Generates n_samples per target (batched)
- Saves all samples to generated_all.xyz
- Saves best per target (lowest Kabsch RMSD) to best/{Sym}{N}_best.xyz

Usage:
    python generate.py \
        --checkpoint checkpoints/best_model.pt \
        --test_data processed_data/test_dataset.pkl \
        --output_dir results/ \
        --n_samples 100 \
        --n_steps 100
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
try:
    from v6_train import NanoparticleCFM
except ImportError:
    raise ImportError("Could not import NanoparticleCFM from v6_train.py.")
try:
    from preprocess_xyz_to_pyg import NanoparticleDataset
except ImportError:
    pass

# ── Element mapping ───────────────────────────────────────────
Z_TO_ELEMENT = {
    1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',
    11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',
    19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',
    27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',34:'Se',
    35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',42:'Mo',
    43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',
    51:'Sb',52:'Te',53:'I',54:'Xe',55:'Cs',56:'Ba',57:'La',58:'Ce',
    59:'Pr',60:'Nd',61:'Pm',62:'Sm',63:'Eu',64:'Gd',65:'Tb',66:'Dy',
    67:'Ho',68:'Er',69:'Tm',70:'Yb',71:'Lu',72:'Hf',73:'Ta',74:'W',
    75:'Re',76:'Os',77:'Ir',78:'Pt',79:'Au',80:'Hg',81:'Tl',82:'Pb',83:'Bi',
}


# ── Kabsch RMSD ───────────────────────────────────────────────

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Returns Kabsch RMSD, or float('inf') if invalid."""
    if not (np.isfinite(P).all() and np.isfinite(Q).all()):
        return float('inf')
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    H = P.T @ Q
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return float('inf')
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    P_rot = P @ R.T
    result = float(np.sqrt(((P_rot - Q) ** 2).sum(axis=1).mean()))
    return result if np.isfinite(result) else float('inf')


# ── ODE integrators ───────────────────────────────────────────

@torch.no_grad()
def get_velocity(model, x, z, batch, t_val, device, n_graphs):
    data = Data(
        z=z.to(device),
        pos=x.to(device),
        batch=batch.to(device),
    )
    t = torch.full((n_graphs,), t_val, device=device, dtype=torch.float32)
    return model(data, t)


def euler_integrate(model, x0, z, batch, n_graphs, n_steps, device,
                    t_start=0.0, t_end=1.0):
    dt = (t_end - t_start) / n_steps
    ts = torch.linspace(t_start, t_end, n_steps + 1)
    x = x0.clone().to(device)
    for i in range(n_steps):
        x = x + dt * get_velocity(model, x, z, batch, ts[i].item(), device, n_graphs)
    return x


def rk4_integrate(model, x0, z, batch, n_graphs, n_steps, device,
                  t_start=0.0, t_end=1.0):
    dt = (t_end - t_start) / n_steps
    ts = torch.linspace(t_start, t_end, n_steps + 1)
    x = x0.clone().to(device)
    for i in range(n_steps):
        t_val = ts[i].item()
        k1 = get_velocity(model, x,             z, batch, t_val,           device, n_graphs)
        k2 = get_velocity(model, x + 0.5*dt*k1, z, batch, t_val + 0.5*dt, device, n_graphs)
        k3 = get_velocity(model, x + 0.5*dt*k2, z, batch, t_val + 0.5*dt, device, n_graphs)
        k4 = get_velocity(model, x + dt*k3,      z, batch, t_val + dt,     device, n_graphs)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x


# ── XYZ writers ──────────────────────────────────────────────

def write_xyz_block(f, pos: np.ndarray, sym: str, comment: str):
    f.write(f"{pos.shape[0]}\n{comment}\n")
    for coord in pos:
        f.write(f"{sym}  {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f}\n")


def save_xyz(pos: np.ndarray, sym: str, filepath: Path, comment: str = ""):
    with open(filepath, 'w') as f:
        write_xyz_block(f, pos, sym, comment)


# ── Main ──────────────────────────────────────────────────────

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved_args = ckpt.get('args', {})

    model = NanoparticleCFM(
        hidden_dim    = saved_args.get('hidden_dim',     256),
        n_interactions= saved_args.get('n_interactions', 4),
        condition_dim = saved_args.get('condition_dim',  128),
        cutoff        = saved_args.get('cutoff',         5.0),
        max_neighbors = saved_args.get('max_neighbors',  128),
        use_gotennet  = saved_args.get('use_gotennet',   True),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    print(f"\nLoading test data: {args.test_data}")
    with open(args.test_data, 'rb') as f:
        test_dataset = pickle.load(f)
    print(f"  {len(test_dataset)} structures in pkl")

    seen, targets = {}, []
    for data in test_dataset:
        n     = data.pos.shape[0]
        z_val = int(data.z[0].item())
        key   = (z_val, n)
        if key not in seen:
            seen[key] = len(targets)
            pos_np = data.pos.numpy().copy()
            pos_np -= pos_np.mean(axis=0)
            targets.append({
                'idx':     len(targets),
                'z_val':   z_val,
                'n_atoms': n,
                'sym':     Z_TO_ELEMENT.get(z_val, f'X{z_val}'),
                'ref_pos': pos_np,
            })
    print(f"  {len(targets)} unique (element, n_atoms) targets")

    output_dir = Path(args.output_dir)
    best_dir   = output_dir / 'best'
    best_dir.mkdir(parents=True, exist_ok=True)

    integrate_fn = euler_integrate if args.integrator == 'euler' else rk4_integrate

    print(f"\nGenerating {args.n_samples} samples per target  "
          f"[{args.integrator}, n_steps={args.n_steps}, max_batch={args.max_batch}]")
    print(f"Total: {len(targets) * args.n_samples} structures\n")

    summary = []
    gen_all_path    = output_dir / 'generated_all.xyz'
    target_all_path = output_dir / 'target_all.xyz'

    # Write target_all.xyz first (one entry per unique target, index-matched)
    with open(target_all_path, 'w') as tgt_f:
        for tgt in targets:
            comment = f"Target structure {tgt['idx']}, element={tgt['z_val']}"
            write_xyz_block(tgt_f, tgt['ref_pos'], tgt['sym'], comment)
    print(f"target_all.xyz → {target_all_path}")

    with open(gen_all_path, 'w') as gen_all_f:
        for tgt in tqdm(targets, desc="Targets"):
            idx     = tgt['idx']
            sym     = tgt['sym']
            z_val   = tgt['z_val']
            n_atoms = tgt['n_atoms']
            ref_pos = tgt['ref_pos']

            best_rmsd  = float('inf')
            best_pos   = None
            remaining  = args.n_samples
            sample_num = 0
            n_fail     = 0

            while remaining > 0:
                c = min(remaining, args.max_batch)
                remaining -= c

                x0_list = []
                for _ in range(c):
                    xi = torch.randn(n_atoms, 3, dtype=torch.float32)
                    xi -= xi.mean(0, keepdim=True)
                    x0_list.append(xi)
                x0    = torch.cat(x0_list, dim=0)
                batch = torch.arange(c, dtype=torch.long).repeat_interleave(n_atoms)
                z     = torch.full((c * n_atoms,), z_val, dtype=torch.long)

                try:
                    x_final = integrate_fn(
                        model, x0, z, batch, c,
                        args.n_steps, device,
                        t_start=args.t_start, t_end=args.t_end,
                    )
                except Exception as e:
                    tqdm.write(f"[WARN] {sym}{n_atoms} exception: {e}")
                    n_fail += c
                    sample_num += c
                    continue

                x_final = x_final.cpu()

                for k in range(c):
                    pos = x_final[k * n_atoms : (k + 1) * n_atoms].numpy()
                    pos -= pos.mean(axis=0)

                    # Skip invalid (nan/inf or exploded)
                    if not np.isfinite(pos).all() or np.abs(pos).max() > 1000:
                        n_fail += 1
                        sample_num += 1
                        continue

                    rmsd = kabsch_rmsd(pos, ref_pos)
                    if not np.isfinite(rmsd):
                        n_fail += 1
                        sample_num += 1
                        continue

                    comment = (f"Target {idx} sample {sample_num} "
                               f"element={sym} n_atoms={n_atoms} kabsch_rmsd={rmsd:.4f}")
                    write_xyz_block(gen_all_f, pos, sym, comment)

                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_pos  = pos.copy()

                    sample_num += 1

            n_valid = args.n_samples - n_fail
            if n_fail > 0:
                tqdm.write(f"  [FAIL] {sym}{n_atoms}: {n_fail}/{args.n_samples} invalid samples")

            if best_pos is not None:
                out_path = best_dir / f"{sym}{n_atoms}_best.xyz"
                save_xyz(best_pos, sym, out_path,
                         comment=f"Target {idx} {sym}{n_atoms} best of {n_valid} valid | kabsch_rmsd={best_rmsd:.4f}")
                summary.append({
                    'target_idx': idx,
                    'target':     f"{sym}{n_atoms}",
                    'element':    sym,
                    'n_atoms':    n_atoms,
                    'best_rmsd':  float(best_rmsd),
                    'n_valid':    n_valid,
                    'n_fail':     n_fail,
                })
            else:
                tqdm.write(f"[WARN] No valid sample for {sym}{n_atoms} (all {n_fail} failed)")

    summary_path = output_dir / 'best_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    total_requested = len(targets) * args.n_samples
    total_fail      = sum(s['n_fail']  for s in summary)
    total_valid     = sum(s['n_valid'] for s in summary)

    print(f"\n{'='*50}")
    print(f"generated_all.xyz → {gen_all_path}")
    print(f"target_all.xyz    → {target_all_path}")
    print(f"Best structures   → {best_dir}/")
    print(f"Summary           → {summary_path}")
    print(f"{'='*50}")
    print(f"Requested : {total_requested}")
    print(f"Valid     : {total_valid}  ({100*total_valid/max(total_requested,1):.1f}%)")
    print(f"Failed    : {total_fail}  ({100*total_fail/max(total_requested,1):.1f}%)")
    print(f"\n{'Target':>12}  {'best_rmsd':>10}  {'valid':>6}  {'fail':>6}")
    print("-" * 42)
    for s in sorted(summary, key=lambda x: x['best_rmsd']):
        print(f"{s['target']:>12}  {s['best_rmsd']:>10.4f}  {s['n_valid']:>6}  {s['n_fail']:>6}")


def main():
    parser = argparse.ArgumentParser(description="Generate nanoparticles with CFM v6")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_data',  type=str, required=True)
    parser.add_argument('--n_samples',  type=int, default=100)
    parser.add_argument('--max_batch',  type=int, default=100)
    parser.add_argument('--n_steps',    type=int, default=100)
    parser.add_argument('--integrator', type=str, default='euler', choices=['euler', 'rk4'])
    parser.add_argument('--t_start',    type=float, default=0.0)
    parser.add_argument('--t_end',      type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='generated/')
    parser.add_argument('--device',     type=str, default='cuda:0')
    args = parser.parse_args()
    generate(args)


if __name__ == '__main__':
    main()
