"""Train LSTMBaseline and EEGGAT on the same synthetic split, compare metrics."""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.data import Batch

from synthetic_data import generate_dataset, band_power_features
from plv import plv_connectivity_matrix, sparsify
from models import LSTMBaseline, EEGGAT, build_graph_data


def prepare_data(n_samples, seed=0, plv_threshold=0.3):
    X, y = generate_dataset(n_samples, seed=seed)
    graphs = []
    for window, label in zip(X, y):
        feats = band_power_features(window)
        feats_norm = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
        adj = sparsify(plv_connectivity_matrix(window), threshold=plv_threshold)
        graphs.append(build_graph_data(
            torch.tensor(feats_norm, dtype=torch.float32),
            torch.tensor(adj, dtype=torch.float32),
            int(label),
        ))
    return X, y, graphs


def train_lstm(X_train, y_train, X_test, y_test, n_channels, epochs):
    model = LSTMBaseline(n_channels=n_channels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()
    return preds, probs


def train_gat(graphs_train, graphs_test, epochs):
    model = EEGGAT()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_batch = Batch.from_data_list(graphs_train)
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(train_batch)
        loss = loss_fn(out, train_batch.y)
        loss.backward()
        opt.step()
    model.eval()
    test_batch = Batch.from_data_list(graphs_test)
    with torch.no_grad():
        logits = model(test_batch)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()
    return preds, probs, model


def report(name, y_true, preds, probs):
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    print(f"{name:15s}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    X, y, graphs = prepare_data(args.n_samples)
    split = int(0.8 * len(y))
    idx = np.random.default_rng(0).permutation(len(y))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    graphs_train = [graphs[i] for i in train_idx]
    graphs_test = [graphs[i] for i in test_idx]

    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}\n")

    preds, probs = train_lstm(X_train, y_train, X_test, y_test, n_channels=X.shape[1], epochs=args.epochs)
    report("LSTMBaseline", y_test, preds, probs)

    preds, probs, model = train_gat(graphs_train, graphs_test, epochs=args.epochs)
    report("EEGGAT", y_test, preds, probs)

    print("\nNote: results are on synthetic data with an injected group")
    print("difference in alpha-band phase coupling — a methods demo, not a")
    print("clinical finding. Swap in real EEG via load_kaggle_eeg() to extend.")


if __name__ == "__main__":
    main()
