"""Train a small PyTorch MLP on the churnml features (CPU only).

Install the CPU build of PyTorch first
(uv add torch --index https://download.pytorch.org/whl/cpu), then run:
    uv run python examples/torch_mlp.py
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churnml.features import build_features, make_preprocessor
from churnml.synth import AS_OF, generate
from churnml.train import split_data

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def to_tensor(matrix: object) -> torch.Tensor:
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return torch.tensor(dense, dtype=torch.float32)


def main() -> None:
    tables = generate(2000, seed=SEED)
    feats = build_features(tables["customers"], tables["usage_monthly"], AS_OF)
    x_tr, x_va, x_te, y_tr, y_va, y_te = split_data(feats, tables["labels"], SEED)

    prep = make_preprocessor().fit(x_tr)  # fit on TRAIN only: no leakage
    xt, xv, xe = (to_tensor(prep.transform(x)) for x in (x_tr, x_va, x_te))
    yt, yv = (torch.tensor(y.to_numpy(), dtype=torch.float32) for y in (y_tr, y_va))

    model = nn.Sequential(
        nn.Linear(xt.shape[1], 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=True)

    best_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(1, 101):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(model(xv).squeeze(1), yv).item()
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, patience = (
                val_loss,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            patience += 1
        if patience >= 10:
            print(f"early stop at epoch {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        proba = torch.sigmoid(model(xe).squeeze(1)).numpy()
    print(
        f"test roc_auc={roc_auc_score(y_te, proba):.3f} "
        f"pr_auc={average_precision_score(y_te, proba):.3f}"
    )

    torch.save(model.state_dict(), "mlp_state.pt")
    reloaded = nn.Sequential(
        nn.Linear(xt.shape[1], 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
    )
    reloaded.load_state_dict(torch.load("mlp_state.pt", weights_only=True))
    reloaded.eval()
    with torch.inference_mode():
        again = torch.sigmoid(reloaded(xe).squeeze(1)).numpy()
    print("reload matches:", bool(np.allclose(proba, again)))


if __name__ == "__main__":
    main()
