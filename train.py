import os
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

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
# RN feature layout (from feature_generators/feature_alignment.py)
# RN is 105 dims:
#   aa_code(20) + ss(3) + depth(1) + properties(7) + sasa(1) + pssm(20) + hhm(30) + msa_freq(21) + cons_score(1) + is_mut(1)
# Indices below are [start:end) Python slices.
# -------------------------
RN_AA = slice(0, 20)
RN_SS = slice(20, 23)
RN_DEPTH = slice(23, 24)
RN_PROP = slice(24, 31)
RN_SASA = slice(31, 32)
RN_PSSM = slice(32, 52)
RN_HHM = slice(52, 82)
RN_MSA_FREQ = slice(82, 103)
RN_MSA_CONS = slice(103, 104)
RN_IS_MUT = slice(104, 105)


# -------------------------
# Ablations
# -------------------------
@dataclass(frozen=True)
class Ablations:
    # coarse modality ablations (existing)
    no_residue_nodes: bool = False    # zero RN (except is_mut flag)
    no_residue_edges: bool = False    # zero RE
    no_atom_nodes: bool = False       # zero AN
    no_atom_edges: bool = False       # zero AE
    no_esm2: bool = False             # zero EF

    # fine-grained RN sub-feature ablations (new)
    no_pssm: bool = False
    no_hhm: bool = False
    no_msa_freq: bool = False
    no_msa_cons: bool = False


def ablation_tag(abl: Ablations) -> str:
    tags = []

    # coarse
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

    # fine-grained (RN content)
    if abl.no_pssm:
        tags.append("no_pssm")
    if abl.no_hhm:
        tags.append("no_hhm")
    if abl.no_msa_freq:
        tags.append("no_msa_freq")
    if abl.no_msa_cons:
        tags.append("no_msa_cons")

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

    def _apply_rn_ablations(self, res_x: np.ndarray, abl: Ablations) -> np.ndarray:
        """
        res_x: (16,105) float array, already standardized for [:,:-1]
        """
        if abl.no_residue_nodes:
            # keep is_mut marker so the model can still locate mutpos
            res_x[:, :-1] = 0.0
            return res_x

        # Fine-grained MSA/evol ablations: these live inside RN.
        # Only apply if we didn't already wipe RN entirely.
        if abl.no_pssm:
            res_x[:, RN_PSSM] = 0.0
        if abl.no_hhm:
            res_x[:, RN_HHM] = 0.0
        if abl.no_msa_freq:
            res_x[:, RN_MSA_FREQ] = 0.0
        if abl.no_msa_cons:
            res_x[:, RN_MSA_CONS] = 0.0

        return res_x

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

        # apply ablations
        res_x = self._apply_rn_ablations(res_x, abl)

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

        # Ablations are applied via dataset attribute set by make_loader()
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

    # coarse modality ablations
    ap.add_argument("--no_residue_nodes", action="store_true",
                    help="Zero-out residue node features RN (except is_mut). This also removes MSA/PSSM/HHM since they live inside RN.")
    ap.add_argument("--no_residue_edges", action="store_true", help="Zero-out residue edge features RE.")
    ap.add_argument("--no_atom_nodes", action="store_true", help="Zero-out atom node features AN.")
    ap.add_argument("--no_atom_edges", action="store_true", help="Zero-out atom edge features AE.")
    ap.add_argument("--no_esm2", action="store_true", help="Zero-out ESM2 residue embeddings EF.")

    # fine-grained RN sub-feature ablations (MSA/evolutionary content)
    ap.add_argument("--no_pssm", action="store_true", help="Zero-out the PSSM block inside RN (dims 32:52).")
    ap.add_argument("--no_hhm", action="store_true", help="Zero-out the HHM block inside RN (dims 52:82).")
    ap.add_argument("--no_msa_freq", action="store_true", help="Zero-out the MSA residue frequency block inside RN (dims 82:103).")
    ap.add_argument("--no_msa_cons", action="store_true", help="Zero-out the MSA conservation score inside RN (dim 103).")

    ap.add_argument("--out-dir", default="runs", help="Directory to write checkpoints and CSVs into.")
    args = ap.parse_args()

    ablations = Ablations(
        no_residue_nodes=args.no_residue_nodes,
        no_residue_edges=args.no_residue_edges,
        no_atom_nodes=args.no_atom_nodes,
        no_atom_edges=args.no_atom_edges,
        no_esm2=args.no_esm2,
        no_pssm=args.no_pssm,
        no_hhm=args.no_hhm,
        no_msa_freq=args.no_msa_freq,
        no_msa_cons=args.no_msa_cons,
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
            f"prefix={prefix}", flush=True
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

    print(f"saved best checkpoint to: {best_ckpt_path} (best_rmse={best_rmse:.4f})",flush=True)
    print(f"saved best test predictions to: {best_test_csv_path}",flush=True)
    print(f"saved last checkpoint to: {last_ckpt_path}",flush=True)


if __name__ == "__main__":
    main()