"""[ARCHIVED — negative result] Phase 1: validate the Mapping Network mechanism on
FashionMNIST from scratch.

RESULT (negative, see phase1_results.txt): a tiny trainable latent z through a FIXED
random mapping net suppressed the train-test gap (every mapping gap 0.8-1.25% < baseline
1.47%; the 27M-param HyperNetwork control overfit worst at 2.55% — so the suppression is
from COMPRESSION, not the architecture) BUT plateaued ~86% test acc (d=4096), ~3pp UNDER
baseline 88.8%, and never reached parity. Lesson: with NO pretrained features to modulate,
the fixed-mapping image is capacity-limited — this is an ELICITATION method, not a
from-scratch trainer. Preserved as the evidence behind the README's "from-scratch is
structurally limited" claim; NOT on the active path.

Core claim tested: a tiny trainable latent z (d ~ hundreds) through a FIXED mapping net
generates a full CNN's weights, matching baseline accuracy AND suppressing the train-test
gap. Key control (HyperNetwork): train the mapping net's weights instead of z — if its gap
matches Mapping's, the suppression is from the architecture, not the compression.

REFACTOR (vs the original repro script): the hardcoded "~/data" download dir is now a
--data-dir flag; no other change (this is a from-scratch CNN, it shares nothing with src/).
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
D = 256          # latent dim z
EPOCHS = 15
torch.manual_seed(0)


# --- target CNN (~50K params), forward takes an explicit flat param vector ---
def cnn_shapes():
    return [("c1.w", (16, 1, 3, 3)), ("c1.b", (16,)), ("c2.w", (32, 16, 3, 3)), ("c2.b", (32,)),
            ("f1.w", (64, 32 * 7 * 7)), ("f1.b", (64,)), ("f2.w", (10, 64)), ("f2.b", (10,))]


P = sum(torch.tensor(s).prod().item() for _, s in cnn_shapes())


def cnn_forward(x, flat):
    i, w = 0, {}
    for n, s in cnn_shapes():
        k = torch.tensor(s).prod().item()
        w[n] = flat[i:i + k].view(s)
        i += k
    x = F.max_pool2d(F.relu(F.conv2d(x, w["c1.w"], w["c1.b"], padding=1)), 2)  # 28->14
    x = F.max_pool2d(F.relu(F.conv2d(x, w["c2.w"], w["c2.b"], padding=1)), 2)  # 14->7
    x = x.flatten(1)
    x = F.relu(F.linear(x, w["f1.w"], w["f1.b"]))
    return F.linear(x, w["f2.w"], w["f2.b"])


# --- the three variants share cnn_forward; they differ in how `flat` is produced ---
class Baseline(nn.Module):                       # train the P params directly
    def __init__(self):
        super().__init__()
        self.flat = nn.Parameter(torch.empty(P).normal_(0, (2.0 / P) ** 0.5))

    def params(self):
        return self.flat


class Mapping(nn.Module):                         # train z; fixed mapping MLP -> flat
    def __init__(self, d=D):
        super().__init__()
        self.z = nn.Parameter(torch.zeros(d).normal_(0, 0.1))
        self.W1 = nn.init.orthogonal_(torch.empty(d, d)).to(DEV)                # fixed
        self.W2 = (torch.empty(P, d).normal_(0, (1.0 / d) ** 0.5)).to(DEV)      # fixed

    def params(self):
        return self.W2 @ torch.tanh(self.W1 @ self.z)


class HyperNet(nn.Module):                        # control: train the mapping, z fixed
    def __init__(self):
        super().__init__()
        self.register_buffer("z", torch.zeros(D).normal_(0, 0.1))               # fixed
        self.W1 = nn.Parameter(nn.init.orthogonal_(torch.empty(D, D)))
        self.W2 = nn.Parameter(torch.empty(P, D).normal_(0, (1.0 / D) ** 0.5))

    def params(self):
        return self.W2 @ torch.tanh(self.W1 @ self.z)


def trainable_count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(name, model, train_dl, test_dl):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEV), y.to(DEV)
            loss = F.cross_entropy(cnn_forward(x, model.params()), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    tr = acc(model, train_dl)
    te = acc(model, test_dl)
    print(f"{name:11s} trainable={trainable_count(model):>9d}  train={tr:.4f}  "
          f"test={te:.4f}  gap={tr-te:.4f}")
    return tr, te


@torch.no_grad()
def acc(model, dl):
    model.eval()
    c = t = 0
    for x, y in dl:
        x, y = x.to(DEV), y.to(DEV)
        c += (cnn_forward(x, model.params()).argmax(1) == y).sum().item()
        t += y.numel()
    return c / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="~/data",
                    help="FashionMNIST download/cache dir (was hardcoded '~/data')")
    args = ap.parse_args()

    tf = transforms.ToTensor()
    train_dl = DataLoader(datasets.FashionMNIST(args.data_dir, True, tf, download=True), 256, shuffle=True)
    test_dl = DataLoader(datasets.FashionMNIST(args.data_dir, False, tf, download=True), 512)
    print(f"device={DEV}  P(target params)={P}  D(latent)={D}  compression={P/D:.0f}x\n")
    run("baseline", Baseline(), train_dl, test_dl)
    for d in [256, 512, 1024, 2048, 4096]:
        run(f"map d={d}", Mapping(d), train_dl, test_dl)
    run("hypernet", HyperNet(), train_dl, test_dl)
    # self-check: param-vector plumbing produces correct shapes
    assert Mapping().to(DEV).params().shape == (P,), "mapping output must be P-dim"


if __name__ == "__main__":
    main()
