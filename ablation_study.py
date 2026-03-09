"""
Ablation study: disable each Bio component one at a time on sklearn digits.
Components: SD Residuals, BCP, BTSP, Adaptive Plateau, Silent Synapses
"""

import numpy as np
import time
from classification_experiments import (
    BackpropClassifier, BioClassifier, load_digits_data, one_hot
)


class AblatedBioClassifier(BioClassifier):
    """BioClassifier with toggles to disable individual components."""

    def __init__(self, sizes, lr=0.1, plateau_gain=3.0, trace_decay=0.7,
                 silent_thresh=3.0, ei_tau=0.05, seed=42,
                 use_sd=True, use_bcp=True, use_btsp=True,
                 use_plateau=True, use_silent=True):
        super().__init__(sizes, lr=lr, plateau_gain=plateau_gain,
                         trace_decay=trace_decay, silent_thresh=silent_thresh,
                         ei_tau=ei_tau, seed=seed)
        self.use_sd = use_sd
        self.use_bcp = use_bcp
        self.use_btsp = use_btsp
        self.use_plateau = use_plateau
        self.use_silent = use_silent

    def train_step(self, x, y_oh):
        a, z = self.forward(x)
        probs = a[-1]
        n = x.shape[0]

        loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))
        error = y_oh - probs

        # Adaptive plateau
        if self.use_plateau:
            err_mag = np.mean(np.abs(error))
            surprise = err_mag / (self.error_ema + 1e-6)
            plateau = 1.0 + (self.plateau_gain - 1.0) * min(surprise, 3.0) / 3.0
            self.error_ema = 0.95 * self.error_ema + 0.05 * err_mag
        else:
            plateau = 1.0

        # SD residuals
        residuals = [None] * self.nl
        residuals[-1] = error / n

        for i in range(self.nl - 2, -1, -1):
            if self.use_sd:
                dendritic_target = error @ self.B[i] / n
            else:
                # No feedback: just use random noise scaled like the signal
                dendritic_target = np.zeros((n, self.sizes[i + 1]))

            if self.use_bcp:
                ei_balance = a[i + 1] - self.I_pred[i]
                self.I_pred[i] += self.ei_tau * np.mean(
                    a[i + 1] - self.I_pred[i], axis=0, keepdims=True)
                bcp_term = 0.01 * ei_balance
            else:
                bcp_term = 0.0

            sd = (dendritic_target + bcp_term) * tanh_deriv(z[i + 1])
            residuals[i] = sd

        # Weight updates
        for i in range(self.nl):
            local_err = residuals[i] * plateau
            dW = a[i].T @ local_err
            db = np.sum(local_err, axis=0, keepdims=True)

            if self.use_btsp:
                self.traces[i] = self.trace_decay * self.traces[i] + (1 - self.trace_decay) * dW
                self.traces_b[i] = self.trace_decay * self.traces_b[i] + (1 - self.trace_decay) * db
                trace_contrib = 0.3 * self.traces[i]
                trace_contrib_b = 0.3 * self.traces_b[i]
            else:
                trace_contrib = 0.0
                trace_contrib_b = 0.0

            if self.use_silent:
                mask = self._silent_mask(self.W[i])
            else:
                mask = 1.0

            self.W[i] += self.lr * (dW + trace_contrib) * mask
            self.b[i] += self.lr * (db + trace_contrib_b)
            np.clip(self.W[i], -5.0, 5.0, out=self.W[i])

        return loss


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def run_ablation(epochs=50, batch_size=32):
    print("Loading digits...")
    X_tr, y_tr, X_te, y_te, nc = load_digits_data()
    Y_tr = one_hot(y_tr, nc)
    arch = [64, 128, 64, nc]
    rng_base = np.random.RandomState(42)

    configs = {
        "Backprop":       None,  # special case
        "Bio (full)":     dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=True),
        "- SD Residuals": dict(use_sd=False, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=True),
        "- BCP":          dict(use_sd=True, use_bcp=False, use_btsp=True, use_plateau=True, use_silent=True),
        "- BTSP":         dict(use_sd=True, use_bcp=True, use_btsp=False, use_plateau=True, use_silent=True),
        "- Plateau":      dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=False, use_silent=True),
        "- Silent Syn":   dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=False),
    }

    results = {}
    for name, cfg in configs.items():
        if cfg is None:
            model = BackpropClassifier(arch, lr=0.2, seed=7)
        else:
            model = AblatedBioClassifier(arch, lr=0.3, plateau_gain=3.0,
                                         trace_decay=0.7, seed=7, **cfg)

        rng = np.random.RandomState(42)
        losses = []
        test_accs = []
        train_accs = []

        for ep in range(1, epochs + 1):
            idx = rng.permutation(len(X_tr))
            ep_loss = 0; nb = 0
            for s in range(0, len(X_tr), batch_size):
                e = min(s + batch_size, len(X_tr))
                bi = idx[s:e]
                ep_loss += model.train_step(X_tr[bi], Y_tr[bi])
                nb += 1
            losses.append(ep_loss / nb)

            if ep % 2 == 0 or ep == 1:
                tr = model.accuracy(X_tr, y_tr)
                te = model.accuracy(X_te, y_te)
                train_accs.append((ep, tr))
                test_accs.append((ep, te))

        final = test_accs[-1][1]
        print(f"  {name:>16s}: test_acc={final:.4f}")
        results[name] = {"losses": losses, "test_accs": test_accs, "train_accs": train_accs}

    return results


def plot_ablation(results):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    styles = {
        "Backprop":       dict(color="#1f77b4", ls="-", lw=2),
        "Bio (full)":     dict(color="#d62728", ls="-", lw=2),
        "- SD Residuals": dict(color="#ff7f0e", ls="--", lw=1.5),
        "- BCP":          dict(color="#2ca02c", ls="--", lw=1.5),
        "- BTSP":         dict(color="#9467bd", ls="--", lw=1.5),
        "- Plateau":      dict(color="#8c564b", ls="--", lw=1.5),
        "- Silent Syn":   dict(color="#e377c2", ls="--", lw=1.5),
    }

    for name, d in results.items():
        s = styles[name]
        ax1.plot(d["losses"], label=name, **s, alpha=0.85)
        eps, accs = zip(*d["test_accs"])
        ax2.plot(eps, accs, label=name, **s, alpha=0.85, marker="o", markersize=2.5)

    ax1.set_title("Ablation — Loss (Digits)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy")
    ax1.set_yscale("log"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.set_title("Ablation — Test Accuracy (Digits)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1.05]); ax2.legend(fontsize=8, loc="lower right"); ax2.grid(True, alpha=0.3)

    # Bar chart of final accuracies
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    finals = [results[n]["test_accs"][-1][1] for n in names]
    colors = [styles[n]["color"] for n in names]
    bars = ax3.barh(names, finals, color=colors, alpha=0.85)
    ax3.set_xlim([0.5, 1.0])
    ax3.set_xlabel("Final Test Accuracy")
    ax3.set_title("Component Ablation — Final Accuracy")
    for bar, val in zip(bars, finals):
        ax3.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    fig.tight_layout()
    fig.savefig("/Users/sbae703/Research/ML/local_max/ablation_curves.png", dpi=150)
    fig2.savefig("/Users/sbae703/Research/ML/local_max/ablation_bars.png", dpi=150)
    print("  Saved ablation_curves.png and ablation_bars.png")
    plt.show()


if __name__ == "__main__":
    results = run_ablation()
    plot_ablation(results)
