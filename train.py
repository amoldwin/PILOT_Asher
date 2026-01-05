import os
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model_0830 import ANTIGEN_18


# -------------------------
# Preprocessing (match predict.py)
# -------------------------
def Normalization(x, scop=1, start=0):
    return scop * (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-8) + start


def Standardization(x):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)


# -------------------------
# Reproducibility helpers
# -------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism knobs (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Mutation parsing
# -------------------------
def mut_id_from_row(pdb: str, chain: str, mut_pos: str, residue_field: str):
    # residue_field like "F/L"
    wt = residue_field[0]
    mt = residue_field[-1]
    return f"{pdb}_{chain}_{wt}{mut_pos}{mt}", wt, mt


# -------------------------
# Ablations
# -------------------------
@dataclass(frozen=True)
class Ablations:
    no_residue_nodes: bool = False    # zero RN (except is_mut flag)
    no_residue_edges: bool = False    # zero RE
    no_atom_nodes: bool = False       # zero AN
    no_atom_edges: bool = False       # zero AE
    no_esm2: bool = False             # zero EF


def ablation_tag(abl: Ablations) -> str:
    tags = []
    if abl.no_residue_nodes:
        tags.append("no_residue_nodes")
    if abl.no_residue_edges:
        tags.append("no_residue_edges")
    if abl.no_atom_nodes:
        tags.append("no_atom_nodes")
    if abl.no_atom_edges:
        tags.append("no_atom_edges")
    if abl.no_esm2:
        tags.append("no_esm2")
    if not tags:
        return "full"
    return "+".join(tags)


def make_prefix(job_id: str, abl: Ablations) -> str:
    return f"{job_id}__{ablation_tag(abl)}"


# -------------------------
# Dataset
# -------------------------
class PilotNPYDataset(Dataset):
    """
    Loads pre-generated PILOT inputs from FEATURE_DIR/input for each mutation in split file.
    Expected split format includes:
      PDB  chain  mut_pos  residue  exp.DDG  pred.DDG
    """
    def __init__(self, split_tsv: str, feature_dir: str, input_subdir: str = "input"):
        self.split_tsv = split_tsv
        self.input_dir = os.path.join(feature_dir, input_subdir)

        self.rows: List[Tuple[str, str, str, str, float]] = []
        with open(split_tsv, "r") as f:
            _header = f.readline()  # consume header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"Bad line in {split_tsv}: {line!r}")
                pdb, chain, mut_pos, residue = parts[0], parts[1], parts[2], parts[3]
                exp_ddg = float(parts[4])
                self.rows.append((pdb, chain, mut_pos, residue, exp_ddg))

    def __len__(self):
        return len(self.rows)

    def _load_one(self, mut_id: str, suffix: str, abl: Ablations):
        """
        suffix in {"wt","mt"}
        Returns tensors in the same order model.forward expects:
        res_x, res_ei, res_e, atom_x, atom_ei, atom_e, extra, index
        """
        base = os.path.join(self.input_dir, f"{mut_id}_")

        res_x = np.load(base + f"RN_{suffix}.npy").astype(float)  # (16, 105)
        res_e = np.load(base + f"RE_{suffix}.npy").astype(float)  # (16, 2)
        res_ei = np.load(base + f"REI_{suffix}.npy").astype(int)  # (2, 16)

        atom_x = np.load(base + f"AN_{suffix}.npy").astype(float)  # (n_atoms, 5)
        atom_e = np.load(base + f"AE_{suffix}.npy").astype(float)  # (n_edges, 3)
        atom_ei = np.load(base + f"AEI_{suffix}.npy").astype(int)  # (2, n_edges)

        # In this repo this is residue boundaries [[start,end], ...] used by AtomPooling
        index = np.load(base + f"I_{suffix}.npy").astype(int)

        extra = np.load(base + f"EF_{suffix}.npy").astype(float)  # (16, 1280)

        # preprocessing identical to predict.py
        res_x[:, :-1] = Standardization(res_x[:, :-1])  # keep last "is_mut" flag
        res_e = Normalization(res_e)
        atom_e = Normalization(atom_e)

        # apply ablations (zero-out modality content)
        if abl.no_residue_nodes:
            # keep the "is_mut" marker so mutpos can still be found
            res_x[:, :-1] = 0.0
        if abl.no_residue_edges:
            res_e[:] = 0.0
        if abl.no_atom_nodes:
            atom_x[:] = 0.0
        if abl.no_atom_edges:
            atom_e[:] = 0.0
        if abl.no_esm2:
            extra[:] = 0.0

        # to torch
        res_x = torch.tensor(res_x, dtype=torch.float32)
        res_e = torch.tensor(res_e, dtype=torch.float32)
        res_ei = torch.tensor(res_ei, dtype=torch.int64)

        atom_x = torch.tensor(atom_x, dtype=torch.float32)
        atom_e = torch.tensor(atom_e, dtype=torch.float32)
        atom_ei = torch.tensor(atom_ei, dtype=torch.int64)

        index = torch.tensor(index, dtype=torch.int64)
        extra = torch.tensor(extra, dtype=torch.float32)

        return res_x, res_ei, res_e, atom_x, atom_ei, atom_e, extra, index

    def __getitem__(self, idx):
        pdb, chain, mut_pos, residue, y = self.rows[idx]
        mut_id, _, _ = mut_id_from_row(pdb, chain, mut_pos, residue)

        # Ablations will be applied in the training loop via a closure-like attribute.
        # We set this attribute from outside (see make_loader()).
        abl: Ablations = getattr(self, "_ablations", Ablations())

        wt = self._load_one(mut_id, "wt", abl)
        mt = self._load_one(mut_id, "mt", abl)

        y = torch.tensor([[y]], dtype=torch.float32)      # (1,1)
        rand = torch.tensor([[0.0]], dtype=torch.float32) # model expects this arg

        meta = {
            "pdb": pdb,
            "chain": chain,
            "mut_pos": mut_pos,
            "residue": residue,
            "mut_id": mut_id,
        }

        return (*wt, *mt, rand, y, meta)


def collate_single(batch):
    """
    Model hard-codes batch_size=1 in attention layers, so enforce bs=1.
    """
    if len(batch) != 1:
        raise ValueError("This training script currently supports batch_size=1 due to model implementation.")
    return batch[0]


def make_loader(ds: PilotNPYDataset, ablations: Ablations, shuffle: bool, seed: int, num_workers: int = 0):
    # Attach ablations to dataset so __getitem__ can apply them
    ds._ablations = ablations  # type: ignore[attr-defined]

    # Make DataLoader shuffling deterministic via generator
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_single,
        generator=g if shuffle else None,
    )


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    n = 0
    loss_fn = torch.nn.MSELoss()

    for sample in loader:
        (res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
         res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
         rand, y, _meta) = sample

        # to device
        res_x_wt = res_x_wt.to(device)
        res_ei_wt = res_ei_wt.to(device)
        res_e_wt = res_e_wt.to(device)
        atom_x_wt = atom_x_wt.to(device)
        atom_ei_wt = atom_ei_wt.to(device)
        atom_e_wt = atom_e_wt.to(device)
        extra_wt = extra_wt.to(device)
        index_wt = index_wt.to(device)

        res_x_mt = res_x_mt.to(device)
        res_ei_mt = res_ei_mt.to(device)
        res_e_mt = res_e_mt.to(device)
        atom_x_mt = atom_x_mt.to(device)
        atom_ei_mt = atom_ei_mt.to(device)
        atom_e_mt = atom_e_mt.to(device)
        extra_mt = extra_mt.to(device)
        index_mt = index_mt.to(device)

        rand = rand.to(device)
        y = y.to(device)

        pred = model(
            res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
            res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
            rand
        )

        loss = loss_fn(pred, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total_loss += float(loss.item())
        n += 1

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate_and_collect(model, loader, device):
    """
    Returns:
      metrics dict
      rows list of dicts for CSV writing
    """
    model.eval()
    ys = []
    ps = []
    csv_rows: List[Dict[str, Any]] = []

    for sample in loader:
        (res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
         res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
         rand, y, meta) = sample

        # to device
        res_x_wt = res_x_wt.to(device)
        res_ei_wt = res_ei_wt.to(device)
        res_e_wt = res_e_wt.to(device)
        atom_x_wt = atom_x_wt.to(device)
        atom_ei_wt = atom_ei_wt.to(device)
        atom_e_wt = atom_e_wt.to(device)
        extra_wt = extra_wt.to(device)
        index_wt = index_wt.to(device)

        res_x_mt = res_x_mt.to(device)
        res_ei_mt = res_ei_mt.to(device)
        res_e_mt = res_e_mt.to(device)
        atom_x_mt = atom_x_mt.to(device)
        atom_ei_mt = atom_ei_mt.to(device)
        atom_e_mt = atom_e_mt.to(device)
        extra_mt = extra_mt.to(device)
        index_mt = index_mt.to(device)

        rand = rand.to(device)

        pred = model(
            res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
            res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
            rand
        )

        y_val = float(y.item())
        p_val = float(pred.detach().cpu().item())

        ys.append(y_val)
        ps.append(p_val)

        csv_rows.append({
            "PDB": meta["pdb"],
            "chain": meta["chain"],
            "mut_pos": meta["mut_pos"],
            "residue": meta["residue"],
            "exp.DDG": y_val,
            "pred.DDG": p_val,
            "mut_id": meta["mut_id"],
        })

    ys = np.array(ys, dtype=float)
    ps = np.array(ps, dtype=float)
    mse = float(np.mean((ps - ys) ** 2)) if len(ys) else float("nan")
    rmse = float(np.sqrt(mse)) if len(ys) else float("nan")
    mae = float(np.mean(np.abs(ps - ys))) if len(ys) else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae}, csv_rows


def write_csv(path: str, rows: List[Dict[str, Any]]):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["PDB", "chain", "mut_pos", "residue", "exp.DDG", "pred.DDG", "mut_id"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train_set.txt")
    ap.add_argument("--test", required=True, help="Path to test_set.txt")
    ap.add_argument("--feature-dir", required=True, help="Root feature dir passed to gen_features.py")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--seed", type=int, default=0,
                    help="Global seed used for model init + deterministic DataLoader shuffling.")
    ap.add_argument("--job-id", type=str, required=True,
                    help="Identifier used to prefix saved artifacts (checkpoints + test CSV).")

    # modality ablations
    ap.add_argument("--no_residue_nodes", action="store_true", help="Zero-out residue node features RN (except is_mut).")
    ap.add_argument("--no_residue_edges", action="store_true", help="Zero-out residue edge features RE.")
    ap.add_argument("--no_atom_nodes", action="store_true", help="Zero-out atom node features AN.")
    ap.add_argument("--no_atom_edges", action="store_true", help="Zero-out atom edge features AE.")
    ap.add_argument("--no_esm2", action="store_true", help="Zero-out ESM2 residue embeddings EF.")

    ap.add_argument("--out-dir", default="runs", help="Directory to write checkpoints and CSVs into.")
    args = ap.parse_args()

    ablations = Ablations(
        no_residue_nodes=args.no_residue_nodes,
        no_residue_edges=args.no_residue_edges,
        no_atom_nodes=args.no_atom_nodes,
        no_atom_edges=args.no_atom_edges,
        no_esm2=args.no_esm2,
    )
    prefix = make_prefix(args.job_id, ablations)

    os.makedirs(args.out_dir, exist_ok=True)

    set_global_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = PilotNPYDataset(args.train, args.feature_dir)
    test_ds = PilotNPYDataset(args.test, args.feature_dir)

    # Deterministic shuffling controlled by seed
    train_loader = make_loader(train_ds, ablations, shuffle=True, seed=args.seed)
    test_loader = make_loader(test_ds, ablations, shuffle=False, seed=args.seed)

    # Model init is now deterministic due to set_global_seed()
    model = ANTIGEN_18().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_rmse = float("inf")
    best_ckpt_path = os.path.join(args.out_dir, f"{prefix}__best.pt")
    last_ckpt_path = os.path.join(args.out_dir, f"{prefix}__last.pt")
    best_test_csv_path = os.path.join(args.out_dir, f"{prefix}__test_predictions.csv")

    for epoch in range(1, args.epochs + 1):
        tr_mse = train_one_epoch(model, train_loader, optim, device)
        metrics, _rows = evaluate_and_collect(model, test_loader, device)

        print(
            f"epoch={epoch:03d} "
            f"train_mse={tr_mse:.4f} "
            f"test_rmse={metrics['rmse']:.4f} "
            f"test_mae={metrics['mae']:.4f} "
            f"prefix={prefix}"
        )

        # Save "last" every epoch (useful for resuming)
        torch.save(
            {
                "epoch": epoch,
                "seed": args.seed,
                "job_id": args.job_id,
                "ablations": ablation_tag(ablations),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "metrics": metrics,
            },
            last_ckpt_path,
        )

        # Save "best" checkpoint + write CSV predictions for the best epoch
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            torch.save(
                {
                    "epoch": epoch,
                    "seed": args.seed,
                    "job_id": args.job_id,
                    "ablations": ablation_tag(ablations),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "metrics": metrics,
                },
                best_ckpt_path,
            )
            # also save test predictions CSV for this best model state
            _best_metrics, best_rows = evaluate_and_collect(model, test_loader, device)
            write_csv(best_test_csv_path, best_rows)

    print(f"saved best checkpoint to: {best_ckpt_path} (best_rmse={best_rmse:.4f})")
    print(f"saved best test predictions to: {best_test_csv_path}")
    print(f"saved last checkpoint to: {last_ckpt_path}")


if __name__ == "__main__":
    main()