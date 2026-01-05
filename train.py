import os
import argparse
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model_0830 import ANTIGEN_18


def Normalization(x, scop=1, start=0):
    return scop * (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-8) + start


def Standardization(x):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mut_id_from_row(pdb: str, chain: str, mut_pos: str, residue_field: str):
    wt = residue_field[0]
    mt = residue_field[-1]
    return f"{pdb}_{chain}_{wt}{mut_pos}{mt}", wt, mt


RN_PSSM = slice(32, 52)
RN_HHM = slice(52, 82)
RN_MSA_FREQ = slice(82, 103)
RN_MSA_CONS = slice(103, 104)


@dataclass(frozen=True)
class Ablations:
    no_residue_nodes: bool = False
    no_residue_edges: bool = False
    no_atom_nodes: bool = False
    no_atom_edges: bool = False
    no_esm2: bool = False

    no_pssm: bool = False
    no_hhm: bool = False
    no_msa_freq: bool = False
    no_msa_cons: bool = False


def ablation_tag(abl: Ablations) -> str:
    tags = []
    if abl.no_residue_nodes: tags.append("no_residue_nodes")
    if abl.no_residue_edges: tags.append("no_residue_edges")
    if abl.no_atom_nodes: tags.append("no_atom_nodes")
    if abl.no_atom_edges: tags.append("no_atom_edges")
    if abl.no_esm2: tags.append("no_esm2")
    if abl.no_pssm: tags.append("no_pssm")
    if abl.no_hhm: tags.append("no_hhm")
    if abl.no_msa_freq: tags.append("no_msa_freq")
    if abl.no_msa_cons: tags.append("no_msa_cons")
    return "full" if not tags else "+".join(tags)


def make_prefix(job_id: str, abl: Ablations) -> str:
    return f"{job_id}__{ablation_tag(abl)}"


def _nan_to_num_inplace(x: np.ndarray) -> int:
    """Replace NaN/Inf with finite values. Returns count of non-finite entries fixed."""
    nonfinite = ~np.isfinite(x)
    n_bad = int(nonfinite.sum())
    if n_bad:
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return n_bad


def _validate_index_list(index_arr: np.ndarray, n_atoms: int) -> bool:
    """
    index_arr is expected to be shape (n_res, 2) with inclusive [start, end] per residue.
    Checks bounds and non-empty slices.
    """
    if index_arr.ndim != 2 or index_arr.shape[1] != 2:
        return False
    if index_arr.shape[0] == 0:
        return False

    starts = index_arr[:, 0].astype(int)
    ends = index_arr[:, 1].astype(int)

    if np.any(starts < 0) or np.any(ends < 0):
        return False
    if np.any(starts > ends):
        return False
    if np.any(ends >= n_atoms):
        return False

    # Optional sanity: mostly non-decreasing (not strictly required but typical)
    if np.any(starts[1:] < starts[:-1] - 5):
        return False

    return True


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; returns nan if undefined."""
    if x.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """
    Minimal 'rankdata' with average ranks for ties (1..n), implemented without scipy.
    """
    a = np.asarray(a)
    n = a.size
    order = np.argsort(a, kind="mergesort")  # stable for ties
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        # average rank for ties in [i, j]
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1

    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via Pearson on ranks; returns nan if undefined."""
    if x.size < 2:
        return float("nan")
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    return _safe_corrcoef(rx, ry)


class PilotNPYDataset(Dataset):
    def __init__(
        self,
        split_tsv: str,
        feature_dir: str,
        input_subdir: str = "input",
        warn_missing: bool = True,
        max_warn_missing: int = 20,
        missing_mut_ids: Optional[Set[str]] = None,
        sanitize: bool = True,
        warn_sanitize: bool = True,
        max_warn_sanitize: int = 20,
    ):
        self.split_tsv = split_tsv
        self.input_dir = os.path.join(feature_dir, input_subdir)

        self.warn_missing = warn_missing
        self.max_warn_missing = max_warn_missing
        self._warned_missing = 0

        self.sanitize = sanitize
        self.warn_sanitize = warn_sanitize
        self.max_warn_sanitize = max_warn_sanitize
        self._warned_sanitize = 0
        self.sanitized_values = 0
        self.bad_index_count = 0

        self.rows: List[Tuple[str, str, str, str, float]] = []
        with open(split_tsv, "r") as f:
            _header = f.readline()
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

        self.missing_count = 0
        self.missing_mut_ids = missing_mut_ids

    def __len__(self):
        return len(self.rows)

    def _apply_rn_ablations(self, res_x: np.ndarray, abl: Ablations) -> np.ndarray:
        if abl.no_residue_nodes:
            res_x[:, :-1] = 0.0
            return res_x
        if abl.no_pssm:
            res_x[:, RN_PSSM] = 0.0
        if abl.no_hhm:
            res_x[:, RN_HHM] = 0.0
        if abl.no_msa_freq:
            res_x[:, RN_MSA_FREQ] = 0.0
        if abl.no_msa_cons:
            res_x[:, RN_MSA_CONS] = 0.0
        return res_x

    def _required_paths(self, mut_id: str, suffix: str) -> List[str]:
        base = os.path.join(self.input_dir, f"{mut_id}_")
        return [
            base + f"RN_{suffix}.npy",
            base + f"RE_{suffix}.npy",
            base + f"REI_{suffix}.npy",
            base + f"AN_{suffix}.npy",
            base + f"AE_{suffix}.npy",
            base + f"AEI_{suffix}.npy",
            base + f"I_{suffix}.npy",
            base + f"EF_{suffix}.npy",
        ]

    def _load_one(self, mut_id: str, suffix: str, abl: Ablations):
        base = os.path.join(self.input_dir, f"{mut_id}_")

        res_x = np.load(base + f"RN_{suffix}.npy").astype(float)
        res_e = np.load(base + f"RE_{suffix}.npy").astype(float)
        res_ei = np.load(base + f"REI_{suffix}.npy").astype(int)

        atom_x = np.load(base + f"AN_{suffix}.npy").astype(float)
        atom_e = np.load(base + f"AE_{suffix}.npy").astype(float)
        atom_ei = np.load(base + f"AEI_{suffix}.npy").astype(int)

        index = np.load(base + f"I_{suffix}.npy").astype(int)
        extra = np.load(base + f"EF_{suffix}.npy").astype(float)

        if not _validate_index_list(index, n_atoms=atom_x.shape[0]):
            raise ValueError(f"Bad atom->res index list for {mut_id} {suffix}: shape={index.shape} n_atoms={atom_x.shape[0]}")

        # preprocessing identical to predict.py
        res_x[:, :-1] = Standardization(res_x[:, :-1])
        res_e = Normalization(res_e)
        atom_e = Normalization(atom_e)

        if self.sanitize:
            fixed = 0
            fixed += _nan_to_num_inplace(res_x)
            fixed += _nan_to_num_inplace(res_e)
            fixed += _nan_to_num_inplace(atom_x)
            fixed += _nan_to_num_inplace(atom_e)
            fixed += _nan_to_num_inplace(extra)
            self.sanitized_values += fixed
            if fixed and self.warn_sanitize and self._warned_sanitize < self.max_warn_sanitize:
                self._warned_sanitize += 1
                warnings.warn(f"[PilotNPYDataset] Sanitized {fixed} non-finite values for mut_id={mut_id} ({suffix}).")
                if self._warned_sanitize == self.max_warn_sanitize:
                    warnings.warn("[PilotNPYDataset] Reached max_warn_sanitize; suppressing further sanitize warnings.")

        # ablations
        res_x = self._apply_rn_ablations(res_x, abl)
        if abl.no_residue_edges:
            res_e[:] = 0.0
        if abl.no_atom_nodes:
            atom_x[:] = 0.0
        if abl.no_atom_edges:
            atom_e[:] = 0.0
        if abl.no_esm2:
            extra[:] = 0.0

        return (
            torch.tensor(res_x, dtype=torch.float32),
            torch.tensor(res_ei, dtype=torch.int64),
            torch.tensor(res_e, dtype=torch.float32),
            torch.tensor(atom_x, dtype=torch.float32),
            torch.tensor(atom_ei, dtype=torch.int64),
            torch.tensor(atom_e, dtype=torch.float32),
            torch.tensor(extra, dtype=torch.float32),
            torch.tensor(index, dtype=torch.int64),
        )

    def __getitem__(self, idx):
        pdb, chain, mut_pos, residue, y = self.rows[idx]
        mut_id, _, _ = mut_id_from_row(pdb, chain, mut_pos, residue)
        abl: Ablations = getattr(self, "_ablations", Ablations())

        missing = []
        for suffix in ("wt", "mt"):
            for p in self._required_paths(mut_id, suffix):
                if not os.path.exists(p):
                    missing.append(p)

        if missing:
            self.missing_count += 1
            if self.missing_mut_ids is not None:
                self.missing_mut_ids.add(mut_id)
            if self.warn_missing and self._warned_missing < self.max_warn_missing:
                self._warned_missing += 1
                warnings.warn(
                    f"[PilotNPYDataset] Missing {len(missing)} input files for mut_id={mut_id}; skipping.\n"
                    + "\n".join(missing[:8])
                    + ("" if len(missing) <= 8 else f"\n... (+{len(missing)-8} more)")
                )
                if self._warned_missing == self.max_warn_missing:
                    warnings.warn("[PilotNPYDataset] Reached max_warn_missing; suppressing further missing-file warnings.")
            return None

        try:
            wt = self._load_one(mut_id, "wt", abl)
            mt = self._load_one(mut_id, "mt", abl)
        except ValueError as e:
            self.bad_index_count += 1
            warnings.warn(f"[PilotNPYDataset] {e}; skipping mut_id={mut_id}")
            return None

        y = torch.tensor([[y]], dtype=torch.float32)
        rand = torch.tensor([[0.0]], dtype=torch.float32)
        meta = {"pdb": pdb, "chain": chain, "mut_pos": mut_pos, "residue": residue, "mut_id": mut_id}
        return (*wt, *mt, rand, y, meta)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    if len(batch) != 1:
        raise ValueError("Expected batch_size=1 (model hard-codes it).")
    return batch[0]


def make_loader(ds: PilotNPYDataset, ablations: Ablations, shuffle: bool, seed: int, num_workers: int = 0):
    ds._ablations = ablations  # type: ignore[attr-defined]
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        generator=g if shuffle else None,
    )


def _tensor_is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def train_one_epoch(model, loader, optim, device):
    model.train()
    loss_fn = torch.nn.MSELoss()

    total_loss = 0.0
    used = 0
    skipped_missing = 0
    skipped_nan = 0

    for sample in loader:
        if sample is None:
            skipped_missing += 1
            continue

        (res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
         res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
         rand, y, meta) = sample

        res_x_wt = res_x_wt.to(device); res_ei_wt = res_ei_wt.to(device); res_e_wt = res_e_wt.to(device)
        atom_x_wt = atom_x_wt.to(device); atom_ei_wt = atom_ei_wt.to(device); atom_e_wt = atom_e_wt.to(device)
        extra_wt = extra_wt.to(device); index_wt = index_wt.to(device)

        res_x_mt = res_x_mt.to(device); res_ei_mt = res_ei_mt.to(device); res_e_mt = res_e_mt.to(device)
        atom_x_mt = atom_x_mt.to(device); atom_ei_mt = atom_ei_mt.to(device); atom_e_mt = atom_e_mt.to(device)
        extra_mt = extra_mt.to(device); index_mt = index_mt.to(device)

        rand = rand.to(device); y = y.to(device)

        pred = model(
            res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
            res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
            rand
        )

        if not _tensor_is_finite(pred):
            skipped_nan += 1
            if skipped_nan <= 5:
                warnings.warn(f"[train] Non-finite prediction for mut_id={meta['mut_id']}; skipping batch.")
            continue

        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            skipped_nan += 1
            if skipped_nan <= 5:
                warnings.warn(f"[train] Non-finite loss for mut_id={meta['mut_id']} (pred={pred.item()} y={y.item()}); skipping batch.")
            continue

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

        total_loss += float(loss.item())
        used += 1

    avg = total_loss / used if used > 0 else float("nan")
    return avg, {"used": used, "skipped_missing": skipped_missing, "skipped_nan": skipped_nan}


@torch.no_grad()
def evaluate_and_collect(model, loader, device):
    model.eval()
    ys = []
    ps = []
    rows: List[Dict[str, Any]] = []

    used = 0
    skipped_missing = 0
    skipped_nan = 0

    for sample in loader:
        if sample is None:
            skipped_missing += 1
            continue

        (res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
         res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
         rand, y, meta) = sample

        res_x_wt = res_x_wt.to(device); res_ei_wt = res_ei_wt.to(device); res_e_wt = res_e_wt.to(device)
        atom_x_wt = atom_x_wt.to(device); atom_ei_wt = atom_ei_wt.to(device); atom_e_wt = atom_e_wt.to(device)
        extra_wt = extra_wt.to(device); index_wt = index_wt.to(device)

        res_x_mt = res_x_mt.to(device); res_ei_mt = res_ei_mt.to(device); res_e_mt = res_e_mt.to(device)
        atom_x_mt = atom_x_mt.to(device); atom_ei_mt = atom_ei_mt.to(device); atom_e_mt = atom_e_mt.to(device)
        extra_mt = extra_mt.to(device); index_mt = index_mt.to(device)

        rand = rand.to(device)

        pred = model(
            res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
            res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
            rand
        )

        if not _tensor_is_finite(pred):
            skipped_nan += 1
            continue

        y_val = float(y.item())
        p_val = float(pred.detach().cpu().item())

        if not np.isfinite(p_val) or not np.isfinite(y_val):
            skipped_nan += 1
            continue

        ys.append(y_val)
        ps.append(p_val)
        rows.append({
            "PDB": meta["pdb"],
            "chain": meta["chain"],
            "mut_pos": meta["mut_pos"],
            "residue": meta["residue"],
            "exp.DDG": y_val,
            "pred.DDG": p_val,
            "mut_id": meta["mut_id"],
        })
        used += 1

    ys = np.array(ys, dtype=float)
    ps = np.array(ps, dtype=float)
    if used == 0:
        metrics = {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
    else:
        mse = float(np.mean((ps - ys) ** 2))
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(ps - ys))),
            "pearson": _safe_corrcoef(ps, ys),
            "spearman": _spearmanr(ps, ys),
        }

    return metrics, rows, {"used": used, "skipped_missing": skipped_missing, "skipped_nan": skipped_nan}


def write_csv(path: str, rows: List[Dict[str, Any]]):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["PDB", "chain", "mut_pos", "residue", "exp.DDG", "pred.DDG", "mut_id"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_lines(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for ln in lines:
            f.write(str(ln) + "\n")


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--feature-dir", required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-id", type=str, required=True)

    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--write-missing", default=None)
    ap.add_argument("--max-warn-missing", type=int, default=20)

    # sanitation toggles
    ap.add_argument("--sanitize", action="store_true", default=True, help="Replace NaN/Inf in loaded arrays with 0 (default on).")
    ap.add_argument("--no-sanitize", dest="sanitize", action="store_false")
    ap.add_argument("--max-warn-sanitize", type=int, default=20)

    # coarse modality ablations
    ap.add_argument("--no_residue_nodes", action="store_true")
    ap.add_argument("--no_residue_edges", action="store_true")
    ap.add_argument("--no_atom_nodes", action="store_true")
    ap.add_argument("--no_atom_edges", action="store_true")
    ap.add_argument("--no_esm2", action="store_true")

    # fine-grained RN ablations
    ap.add_argument("--no_pssm", action="store_true")
    ap.add_argument("--no_hhm", action="store_true")
    ap.add_argument("--no_msa_freq", action="store_true")
    ap.add_argument("--no_msa_cons", action="store_true")

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

    print("=" * 88, flush=True)
    print("PILOT training script", flush=True)
    print(f"job_id        : {args.job_id}", flush=True)
    print(f"prefix        : {prefix}", flush=True)
    print(f"seed          : {args.seed}", flush=True)
    print(f"device        : {device}", flush=True)
    print(f"train file    : {args.train}", flush=True)
    print(f"test file     : {args.test}", flush=True)
    print(f"feature_dir   : {args.feature_dir}", flush=True)
    print(f"out_dir       : {args.out_dir}", flush=True)
    print(f"epochs        : {args.epochs}", flush=True)
    print(f"lr            : {args.lr}", flush=True)
    print(f"sanitize      : {args.sanitize}", flush=True)
    print(f"ablations     : {ablation_tag(ablations)}", flush=True)
    print("=" * 88, flush=True)

    missing_mut_ids: Set[str] = set()

    print("[data] Loading datasets...", flush=True)
    train_ds = PilotNPYDataset(
        args.train, args.feature_dir,
        max_warn_missing=args.max_warn_missing,
        missing_mut_ids=missing_mut_ids,
        sanitize=args.sanitize,
        max_warn_sanitize=args.max_warn_sanitize,
    )
    test_ds = PilotNPYDataset(
        args.test, args.feature_dir,
        max_warn_missing=args.max_warn_missing,
        missing_mut_ids=missing_mut_ids,
        sanitize=args.sanitize,
        max_warn_sanitize=args.max_warn_sanitize,
    )
    print(f"[data] Train rows: {len(train_ds)} | Test rows: {len(test_ds)}", flush=True)

    print("[data] Building loaders (batch_size=1 enforced)...", flush=True)
    train_loader = make_loader(train_ds, ablations, shuffle=True, seed=args.seed)
    test_loader = make_loader(test_ds, ablations, shuffle=False, seed=args.seed)

    print("[model] Initializing model...", flush=True)
    model = ANTIGEN_18().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params: total={n_params:,} trainable={n_trainable:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print("[train] Starting training loop...", flush=True)

    best_rmse = float("inf")
    best_ckpt_path = os.path.join(args.out_dir, f"{prefix}__best.pt")
    last_ckpt_path = os.path.join(args.out_dir, f"{prefix}__last.pt")
    best_test_csv_path = os.path.join(args.out_dir, f"{prefix}__test_predictions.csv")

    header = (
        f"{'epoch':>5} | {'train_mse':>10} | "
        f"{'test_rmse':>9} {'test_mae':>9} {'pearson':>9} {'spearman':>9} | "
        f"{'train used/miss/nan':>18} | {'test used/miss/nan':>17} | {'best_rmse':>9}"
    )
    print("-" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for epoch in range(1, args.epochs + 1):
        tr_mse, tr_counts = train_one_epoch(model, train_loader, optim, device)
        te_metrics, te_rows, te_counts = evaluate_and_collect(model, test_loader, device)

        line = (
            f"{epoch:5d} | {_fmt(tr_mse):>10} | "
            f"{_fmt(te_metrics['rmse']):>9} {_fmt(te_metrics['mae']):>9} "
            f"{_fmt(te_metrics['pearson']):>9} {_fmt(te_metrics['spearman']):>9} | "
            f"{tr_counts['used']:5d}/{tr_counts['skipped_missing']:4d}/{tr_counts['skipped_nan']:3d} | "
            f"{te_counts['used']:5d}/{te_counts['skipped_missing']:4d}/{te_counts['skipped_nan']:3d} | "
            f"{_fmt(best_rmse):>9}"
        )
        print(line, flush=True)

        torch.save(
            {
                "epoch": epoch,
                "seed": args.seed,
                "job_id": args.job_id,
                "ablations": ablation_tag(ablations),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "metrics": te_metrics,
                "counts": {"train": tr_counts, "test": te_counts},
            },
            last_ckpt_path,
        )

        if te_metrics["rmse"] < best_rmse:
            best_rmse = te_metrics["rmse"]
            torch.save(
                {
                    "epoch": epoch,
                    "seed": args.seed,
                    "job_id": args.job_id,
                    "ablations": ablation_tag(ablations),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "metrics": te_metrics,
                    "counts": {"train": tr_counts, "test": te_counts},
                },
                best_ckpt_path,
            )
            write_csv(best_test_csv_path, te_rows)

    print("-" * len(header), flush=True)
    print("[done] Training complete.", flush=True)
    print(f"[done] saved best checkpoint to: {best_ckpt_path} (best_rmse={best_rmse:.4f})", flush=True)
    print(f"[done] saved best test predictions to: {best_test_csv_path}", flush=True)
    print(f"[done] saved last checkpoint to: {last_ckpt_path}", flush=True)
    print(f"[done] dataset missing counts: train={train_ds.missing_count} test={test_ds.missing_count}", flush=True)
    print(f"[done] sanitized_values: train_ds={train_ds.sanitized_values} test_ds={test_ds.sanitized_values}", flush=True)
    print(f"[done] bad_index: train_ds={train_ds.bad_index_count} test_ds={test_ds.bad_index_count}", flush=True)
    print(f"[done] missing mut_id count (union): {len(missing_mut_ids)}", flush=True)

    if args.write_missing is not None:
        write_lines(args.write_missing, sorted(missing_mut_ids))
        print(f"[done] wrote missing mut_ids to: {args.write_missing}", flush=True)


if __name__ == "__main__":
    main()