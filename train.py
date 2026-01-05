import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model_0830 import ANTIGEN_18


# Match predict.py preprocessing
def Normalization(x, scop=1, start=0):
    return scop * (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-8) + start

def Standardization(x):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)


def mut_id_from_row(pdb, chain, mut_pos, residue_field):
    # residue_field like "F/L"
    wt = residue_field[0]
    mt = residue_field[-1]
    return f"{pdb}_{chain}_{wt}{mut_pos}{mt}", wt, mt


class PilotNPYDataset(Dataset):
    """
    Loads pre-generated PILOT inputs from FEATURE_DIR/input for each mutation in split file.
    Expected split format (tab-separated) includes:
    PDB  chain  mut_pos  residue  exp.DDG  pred.DDG
    """
    def __init__(self, split_tsv, feature_dir, input_subdir="input", training=True):
        self.split_tsv = split_tsv
        self.input_dir = os.path.join(feature_dir, input_subdir)
        self.training = training

        self.rows = []
        with open(split_tsv, "r") as f:
            header = f.readline().strip().split()
            # tolerate either tabs or spaces
            # We assume first line is header like: PDB chain mut_pos residue exp.DDG pred.DDG
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

    def _load_one(self, mut_id, suffix):
        # suffix in {"wt","mt"}
        base = os.path.join(self.input_dir, f"{mut_id}_")
        res_x = np.load(base + f"RN_{suffix}.npy").astype(float)
        res_e = np.load(base + f"RE_{suffix}.npy").astype(float)
        res_ei = np.load(base + f"REI_{suffix}.npy").astype(int)

        atom_x = np.load(base + f"AN_{suffix}.npy").astype(float)
        atom_e = np.load(base + f"AE_{suffix}.npy").astype(float)
        atom_ei = np.load(base + f"AEI_{suffix}.npy").astype(int)

        # Note: In predict.py this is called atom2res_index_* but it is actually residue_index boundaries
        index = np.load(base + f"I_{suffix}.npy").astype(int)

        extra = np.load(base + f"EF_{suffix}.npy").astype(float)

        # Apply same preprocessing as predict.py
        res_x[:, :-1] = Standardization(res_x[:, :-1])   # keep last "is_mut" flag unstandardized
        res_e = Normalization(res_e)
        atom_e = Normalization(atom_e)

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

        wt = self._load_one(mut_id, "wt")
        mt = self._load_one(mut_id, "mt")

        y = torch.tensor([[y]], dtype=torch.float32)  # shape (1,1) to match model output
        rand = torch.tensor([[0.0]], dtype=torch.float32)  # model expects this arg; currently unused

        return (*wt, *mt, rand, y)


def collate_single(batch):
    """
    Current model seems written effectively for batch_size=1 (hard-coded batch_size=1 in attention layers).
    So we enforce batch size 1 and just unwrap.
    """
    if len(batch) != 1:
        raise ValueError("This training script currently supports batch_size=1 due to model implementation.")
    return batch[0]


def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    n = 0

    loss_fn = torch.nn.MSELoss()

    for sample in loader:
        # unpack
        (res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
         res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
         rand, y) = sample

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
def evaluate(model, loader, device):
    model.eval()
    ys = []
    ps = []

    for sample in loader:
        (res_x_wt, res_ei_wt, res_e_wt, atom_x_wt, atom_ei_wt, atom_e_wt, extra_wt, index_wt,
         res_x_mt, res_ei_mt, res_e_mt, atom_x_mt, atom_ei_mt, atom_e_mt, extra_mt, index_mt,
         rand, y) = sample

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

        ys.append(float(y.item()))
        ps.append(float(pred.detach().cpu().item()))

    ys = np.array(ys, dtype=float)
    ps = np.array(ps, dtype=float)
    mse = float(np.mean((ps - ys) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(ps - ys)))
    return {"mse": mse, "rmse": rmse, "mae": mae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train_set.txt")
    ap.add_argument("--test", required=True, help="Path to test_set.txt")
    ap.add_argument("--feature-dir", required=True, help="Root feature dir passed to gen_features.py")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="pilot_trained.pt")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = PilotNPYDataset(args.train, args.feature_dir)
    test_ds = PilotNPYDataset(args.test, args.feature_dir)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_single)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_single)

    model = ANTIGEN_18().to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_rmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device)
        metrics = evaluate(model, test_loader, device)

        print(f"epoch={epoch:03d} train_mse={tr_loss:.4f} test_rmse={metrics['rmse']:.4f} test_mae={metrics['mae']:.4f}")

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            torch.save(model.state_dict(), args.out)

    print(f"saved best model to {args.out} (best_rmse={best_rmse:.4f})")


if __name__ == "__main__":
    main()