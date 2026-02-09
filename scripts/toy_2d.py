import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_circles, make_moons
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, feat_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class LowRankAdapter(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(torch.relu(self.down(x)))


class AdapterClassifier(nn.Module):
    def __init__(self, feature: FeatureMLP, rank: int, num_classes: int = 2):
        super().__init__()
        self.feature = feature
        self.adapter = LowRankAdapter(32, rank)
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        h = self.feature(x)
        h = h + self.adapter(h)
        return self.head(h)


def train(model, loader, epochs=50, lr=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    return model


def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def make_loader(X, y, batch=256):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch, shuffle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="paper/figs")
    args = parser.parse_args()

    # Base task: moons
    X_m, y_m = make_moons(n_samples=2000, noise=0.15, random_state=0)
    X_m = (X_m - X_m.mean(0)) / X_m.std(0)
    X_m_train, X_m_test = X_m[:1500], X_m[1500:]
    y_m_train, y_m_test = y_m[:1500], y_m[1500:]

    feature = FeatureMLP()
    base_model = nn.Sequential(feature, nn.Linear(32, 2))
    base_model = train(base_model, make_loader(X_m_train, y_m_train), epochs=80, lr=1e-2)

    # Freeze feature map
    for p in feature.parameters():
        p.requires_grad = False

    # Target task: circles
    X_c, y_c = make_circles(n_samples=2000, noise=0.1, factor=0.5, random_state=1)
    X_c = (X_c - X_c.mean(0)) / X_c.std(0)
    X_c_train, X_c_test = X_c[:1500], X_c[1500:]
    y_c_train, y_c_test = y_c[:1500], y_c[1500:]

    ranks = [1, 2, 4, 8, 16, 32]
    accs = []
    for r in ranks:
        model = AdapterClassifier(feature, rank=r)
        # Only train adapter + head
        for p in model.feature.parameters():
            p.requires_grad = False
        model = train(model, make_loader(X_c_train, y_c_train), epochs=80, lr=5e-3)
        acc = evaluate(model, make_loader(X_c_test, y_c_test))
        accs.append(acc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(ranks, accs, marker="o")
    plt.xlabel("Adapter rank")
    plt.ylabel("Test Acc (circles)")
    plt.title("Toy low-rank adaptation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "toy_rank.pdf")

    # Save CSV for table/inspection
    csv_path = Path("results") / "toy_rank.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rank,acc\n")
        for r, a in zip(ranks, accs):
            f.write(f"{r},{a:.4f}\n")

    print("Toy rank sweep saved to", csv_path)


if __name__ == "__main__":
    main()
