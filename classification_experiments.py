"""
Classification Benchmarks: Bio vs Backprop

Datasets:
  1. sklearn digits (8x8, 10 classes, 1797 samples) — fast
  2. MNIST (28x28, 10 classes, 70k samples) — real benchmark

Models:
  - Backprop: standard MLP, softmax + cross-entropy, SGD
  - Bio: SD Residuals + BCP + BTSP + silent synapses + adaptive plateau
"""

import numpy as np
import argparse
import time


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def tanh_act(x): return np.tanh(x)
def tanh_deriv(x): return 1.0 - np.tanh(x)**2

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def one_hot(y, k):
    n = len(y)
    oh = np.zeros((n, k))
    oh[np.arange(n), y.astype(int)] = 1.0
    return oh


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_digits_data():
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data / 16.0  # normalize to [0, 1]
    y = d.target.astype(int)
    # 80/20 split
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    return (X[idx[:split]], y[idx[:split]],
            X[idx[split:]], y[idx[split:]], 10)


def load_mnist_data():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(float) / 255.0
        y = mnist.target.astype(int)
    except Exception:
        # Fallback to torch
        import torch
        from torchvision import datasets
        train = datasets.MNIST('/tmp/mnist', train=True, download=True)
        test = datasets.MNIST('/tmp/mnist', train=False, download=True)
        X = np.vstack([train.data.numpy().reshape(-1, 784),
                       test.data.numpy().reshape(-1, 784)]) / 255.0
        y = np.concatenate([train.targets.numpy(), test.targets.numpy()])

    idx = np.random.RandomState(42).permutation(len(X))
    split = 60000
    return (X[idx[:split]], y[idx[:split]],
            X[idx[split:]], y[idx[split:]], 10)


# ---------------------------------------------------------------------------
# Backprop MLP (softmax + cross-entropy)
# ---------------------------------------------------------------------------

class BackpropClassifier:
    def __init__(self, sizes, lr=0.1, seed=42):
        rng = np.random.RandomState(seed)
        self.lr = lr
        self.nl = len(sizes) - 1
        self.W = [rng.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0/sizes[i])
                  for i in range(self.nl)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(self.nl)]

    def forward(self, x):
        a = [x]; z = [x]
        for i in range(self.nl):
            pre = a[-1] @ self.W[i] + self.b[i]
            z.append(pre)
            if i < self.nl - 1:
                a.append(tanh_act(pre))
            else:
                a.append(softmax(pre))
        return a, z

    def train_step(self, x, y_oh):
        a, z = self.forward(x)
        probs = a[-1]
        n = x.shape[0]

        # Cross-entropy loss
        loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))

        # Softmax + CE gradient: probs - targets
        d = (probs - y_oh) / n
        for i in reversed(range(self.nl)):
            self.W[i] -= self.lr * (a[i].T @ d)
            self.b[i] -= self.lr * np.sum(d, axis=0, keepdims=True)
            if i > 0:
                d = (d @ self.W[i].T) * tanh_deriv(z[i])

        return loss

    def predict(self, x):
        return self.forward(x)[0][-1]

    def accuracy(self, x, y):
        pred = np.argmax(self.predict(x), axis=1)
        return np.mean(pred == y)


# ---------------------------------------------------------------------------
# Bio Classifier (SD Residuals + BCP + BTSP + silent synapses)
# ---------------------------------------------------------------------------

class BioClassifier:
    """
    Three-factor learning rule for classification.
    Same mechanisms as BioNetwork from experiments.py, adapted for softmax output.

    Key biological mechanisms:
      - SD Residual: target (one-hot) projected through feedback B to dendrites
      - BCP: E/I balance disruption encodes local error
      - BTSP: eligibility traces for temporal credit (within mini-batch sequences)
      - Adaptive plateau: larger errors → stronger calcium spike → faster learning
      - Silent synapse gating: preferential plasticity on weak synapses
    """

    def __init__(self, sizes, lr=0.1, plateau_gain=3.0, trace_decay=0.7,
                 silent_thresh=3.0, ei_tau=0.05, seed=42):
        rng = np.random.RandomState(seed)
        self.lr = lr
        self.plateau_gain = plateau_gain
        self.trace_decay = trace_decay
        self.silent_thresh = silent_thresh
        self.ei_tau = ei_tau
        self.nl = len(sizes) - 1
        self.sizes = sizes

        # Forward weights
        self.W = [rng.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0/sizes[i])
                  for i in range(self.nl)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(self.nl)]

        # Feedback connections: project from output to each hidden layer
        out = sizes[-1]
        self.B = [rng.randn(out, sizes[i+1]) * np.sqrt(1.0/out)
                  for i in range(self.nl - 1)]

        # BCP: inhibitory prediction per layer
        self.I_pred = [np.zeros((1, sizes[i+1])) for i in range(self.nl)]

        # BTSP eligibility traces
        self.traces = [np.zeros_like(w) for w in self.W]
        self.traces_b = [np.zeros_like(b) for b in self.b]

        # Adaptive plateau: running error magnitude
        self.error_ema = 0.5

    def _silent_mask(self, W):
        return 1.0 / (1.0 + np.abs(W) / self.silent_thresh)

    def forward(self, x):
        a = [x]; z = [x]
        for i in range(self.nl):
            pre = a[-1] @ self.W[i] + self.b[i]
            z.append(pre)
            if i < self.nl - 1:
                a.append(tanh_act(pre))
            else:
                a.append(softmax(pre))
        return a, z

    def train_step(self, x, y_oh):
        a, z = self.forward(x)
        probs = a[-1]
        n = x.shape[0]

        # Cross-entropy loss
        loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))

        # Output error: target - prediction (biological direction)
        error = y_oh - probs

        # Adaptive plateau: surprise-based gain
        err_mag = np.mean(np.abs(error))
        surprise = err_mag / (self.error_ema + 1e-6)
        plateau = 1.0 + (self.plateau_gain - 1.0) * min(surprise, 3.0) / 3.0
        self.error_ema = 0.95 * self.error_ema + 0.05 * err_mag

        # Compute SD residuals at each layer
        residuals = [None] * self.nl

        # Output layer: direct error (softmax gradient direction)
        residuals[-1] = error / n

        # Hidden layers: project output error through feedback B
        for i in range(self.nl - 2, -1, -1):
            # Apical dendritic input: output error projected through feedback
            dendritic_target = error @ self.B[i] / n

            # BCP: E/I balance error
            ei_balance = a[i+1] - self.I_pred[i]
            self.I_pred[i] += self.ei_tau * np.mean(
                a[i+1] - self.I_pred[i], axis=0, keepdims=True)

            # SD Residual = dendritic signal + small balance correction
            # BCP coefficient must be small relative to dendritic signal
            sd = (dendritic_target + 0.01 * ei_balance) * tanh_deriv(z[i+1])
            residuals[i] = sd

        # Three-factor update at each layer
        for i in range(self.nl):
            local_err = residuals[i] * plateau

            # Hebbian: pre × post-error
            dW = a[i].T @ local_err
            db = np.sum(local_err, axis=0, keepdims=True)

            # Eligibility trace accumulation (BTSP)
            self.traces[i] = self.trace_decay * self.traces[i] + (1 - self.trace_decay) * dW
            self.traces_b[i] = self.trace_decay * self.traces_b[i] + (1 - self.trace_decay) * db

            # Silent synapse mask
            mask = self._silent_mask(self.W[i])

            # Update: lr × (gradient + trace reinforcement) × mask
            self.W[i] += self.lr * (dW + 0.3 * self.traces[i]) * mask
            self.b[i] += self.lr * (db + 0.3 * self.traces_b[i])

            # Clip weights to prevent numerical issues on large networks
            np.clip(self.W[i], -5.0, 5.0, out=self.W[i])

        return loss

    def predict(self, x):
        return self.forward(x)[0][-1]

    def accuracy(self, x, y):
        pred = np.argmax(self.predict(x), axis=1)
        return np.mean(pred == y)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_eval(name, model, X_train, y_train, X_test, y_test,
                   n_classes, epochs, batch_size, eval_every):
    Y_train = one_hot(y_train, n_classes)
    n = len(X_train)
    rng = np.random.RandomState(42)

    train_losses = []
    test_accs = []
    train_accs = []
    epoch_times = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        # Shuffle
        idx = rng.permutation(n)
        ep_loss = 0; n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bi = idx[start:end]
            l = model.train_step(X_train[bi], Y_train[bi])
            ep_loss += l; n_batches += 1

        ep_loss /= n_batches
        train_losses.append(ep_loss)

        if ep % eval_every == 0 or ep == 1:
            tr_acc = model.accuracy(X_train, y_train)
            te_acc = model.accuracy(X_test, y_test)
            train_accs.append((ep, tr_acc))
            test_accs.append((ep, te_acc))
            elapsed = time.time() - t0
            epoch_times.append((ep, elapsed))
            print(f"  [{name:>10}] ep {ep:4d}  loss={ep_loss:.4f}  "
                  f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  ({elapsed:.1f}s)")

    return {
        "losses": train_losses,
        "test_accs": test_accs,
        "train_accs": train_accs,
        "epoch_times": epoch_times,
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_digits(epochs=50, batch_size=32, eval_every=2):
    print("Loading sklearn digits (8x8, 10 classes)...")
    X_tr, y_tr, X_te, y_te, nc = load_digits_data()
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, batch_size={batch_size}")

    models = {
        "Backprop": BackpropClassifier([64, 128, 64, nc], lr=0.2, seed=7),
        "Bio":      BioClassifier([64, 128, 64, nc], lr=0.3, plateau_gain=3.0,
                                   trace_decay=0.7, seed=7),
    }

    results = {}
    for name, model in models.items():
        results[name] = train_and_eval(
            name, model, X_tr, y_tr, X_te, y_te, nc,
            epochs, batch_size, eval_every)
        print()
    return results


def run_mnist(epochs=30, batch_size=64, eval_every=1):
    print("Loading MNIST (28x28, 10 classes)...")
    X_tr, y_tr, X_te, y_te, nc = load_mnist_data()
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, batch_size={batch_size}")

    models = {
        "Backprop": BackpropClassifier([784, 256, nc], lr=0.2, seed=7),
        "Bio":      BioClassifier([784, 256, nc], lr=0.2, plateau_gain=3.0,
                                   trace_decay=0.7, seed=7),
    }

    results = {}
    for name, model in models.items():
        results[name] = train_and_eval(
            name, model, X_tr, y_tr, X_te, y_te, nc,
            epochs, batch_size, eval_every)
        print()
    return results


def run_robustness():
    """Test both models at fixed lr across batch sizes — Bio should be more robust."""
    print("Robustness: fixed lr across batch sizes")
    X_tr, y_tr, X_te, y_te, nc = load_digits_data()

    fixed_lr = 0.3  # same lr for both
    results = {"Bio": {}, "Backprop": {}}

    for bs in [4, 8, 16, 32, 64, 128]:
        for name, ModelClass, kwargs in [
            ("Backprop", BackpropClassifier, dict(lr=fixed_lr)),
            ("Bio", BioClassifier, dict(lr=fixed_lr, plateau_gain=3.0,
                                         trace_decay=0.7, silent_thresh=3.0)),
        ]:
            model = ModelClass([64, 128, 64, nc], seed=7, **kwargs)
            rng = np.random.RandomState(42)
            f90 = None
            for ep in range(1, 51):
                idx = rng.permutation(len(X_tr))
                for s in range(0, len(X_tr), bs):
                    e = min(s + bs, len(X_tr)); bi = idx[s:e]
                    model.train_step(X_tr[bi], one_hot(y_tr[bi], nc))
                acc = model.accuracy(X_te, y_te)
                if f90 is None and acc >= 0.90:
                    f90 = ep
            results[name][bs] = {"acc": acc, "first_90": f90}
            f90s = str(f90) if f90 else "never"
            print(f"  [{name:>10}] bs={bs:3d}  test={acc:.3f}  first_90={f90s:>5}")
        print()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_res, robustness_res=None):
    import matplotlib.pyplot as plt

    colors = {"Backprop": "#1f77b4", "Bio": "#d62728"}

    # Count panels: 2 rows per classification exp + 1 for robustness
    class_exps = {k: v for k, v in all_res.items() if k != "Robustness"}
    n_class = len(class_exps)
    has_robust = robustness_res is not None
    n_cols = n_class + (1 if has_robust else 0)

    fig, axes = plt.subplots(2, max(n_cols, 1), figsize=(6 * max(n_cols, 1), 8))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Classification experiments
    for col, (exp_name, res) in enumerate(class_exps.items()):
        ax_loss = axes[0, col]
        ax_acc = axes[1, col]

        for model_name, d in res.items():
            c = colors.get(model_name)
            ax_loss.plot(d["losses"], label=model_name, alpha=0.8, color=c)
            eps, accs = zip(*d["test_accs"])
            ax_acc.plot(eps, accs, label=model_name, alpha=0.8, color=c,
                        marker='o', markersize=3)
            eps_tr, accs_tr = zip(*d["train_accs"])
            ax_acc.plot(eps_tr, accs_tr, '--', alpha=0.4, color=c)

        ax_loss.set_title(f"{exp_name} — Loss")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-Entropy")
        ax_loss.set_yscale("log"); ax_loss.legend(fontsize=9); ax_loss.grid(True, alpha=0.3)
        ax_acc.set_title(f"{exp_name} — Accuracy")
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
        ax_acc.legend(fontsize=9, loc="lower right"); ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim([0, 1.05])

    # Robustness plot
    if has_robust:
        col = n_class
        ax_acc_r = axes[0, col]
        ax_f90 = axes[1, col]

        for model_name, bs_data in robustness_res.items():
            c = colors.get(model_name)
            batch_sizes = sorted(bs_data.keys())
            accs = [bs_data[bs]["acc"] for bs in batch_sizes]
            f90s = [bs_data[bs]["first_90"] if bs_data[bs]["first_90"] else 50
                    for bs in batch_sizes]
            ax_acc_r.plot(batch_sizes, accs, '-o', label=model_name, color=c, markersize=5)
            ax_f90.plot(batch_sizes, f90s, '-o', label=model_name, color=c, markersize=5)

        ax_acc_r.set_title("Robustness — Accuracy vs Batch Size")
        ax_acc_r.set_xlabel("Batch Size"); ax_acc_r.set_ylabel("Test Accuracy")
        ax_acc_r.set_xscale("log", base=2); ax_acc_r.legend(fontsize=9)
        ax_acc_r.grid(True, alpha=0.3); ax_acc_r.set_ylim([0, 1.05])
        ax_acc_r.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5, label='random')

        ax_f90.set_title("Robustness — Epochs to 90%")
        ax_f90.set_xlabel("Batch Size"); ax_f90.set_ylabel("Epochs (50=never)")
        ax_f90.set_xscale("log", base=2); ax_f90.legend(fontsize=9)
        ax_f90.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/Users/sbae703/Research/ML/local_max/classification_results.png", dpi=150)
    print("  Saved classification_results.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default="all",
                    choices=["all", "digits", "mnist", "robustness"])
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    R = {}
    rob_res = None

    if args.experiment in ("all", "digits"):
        print("=" * 60); print("DIGITS (8x8)"); print("=" * 60)
        R["Digits"] = run_digits()

    if args.experiment in ("all", "mnist"):
        print("=" * 60); print("MNIST (28x28)"); print("=" * 60)
        R["MNIST"] = run_mnist()

    if args.experiment in ("all", "robustness"):
        print("=" * 60); print("ROBUSTNESS"); print("=" * 60)
        rob_res = run_robustness()

    # Summary
    print("=" * 60); print("SUMMARY"); print("=" * 60)
    for exp, res in R.items():
        print(f"\n  {exp}:")
        for m, d in res.items():
            final_test = d["test_accs"][-1][1]
            final_train = d["train_accs"][-1][1]
            first_90 = None
            for ep, acc in d["test_accs"]:
                if acc >= 0.90:
                    first_90 = ep; break
            f90 = f"ep {first_90}" if first_90 else "never"
            print(f"    {m:>10}: test={final_test:.4f}  train={final_train:.4f}  first_90%={f90}")

    if rob_res:
        print(f"\n  Robustness (fixed lr=0.3, 50 epochs):")
        for bs in sorted(list(rob_res["Bio"].keys())):
            bp = rob_res["Backprop"][bs]
            bio = rob_res["Bio"][bs]
            bp_f = str(bp["first_90"]) if bp["first_90"] else "never"
            bio_f = str(bio["first_90"]) if bio["first_90"] else "never"
            print(f"    bs={bs:3d}  BP: {bp['acc']:.3f} (90%@{bp_f:>5})  "
                  f"Bio: {bio['acc']:.3f} (90%@{bio_f:>5})")

    if not args.no_plot and (R or rob_res):
        plot_results(R, rob_res)


if __name__ == "__main__":
    main()
