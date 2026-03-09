"""
Simplicity experiment: find the minimal set of bio components that works
across BOTH classification (digits) and temporal/RL (chain, cartpole) tasks.

Tests:
  1. Single-component ablations (remove one at a time)
  2. Minimal combinations (keep only 1-2 components)
  3. Across both task types

Metrics: final performance AND time-to-acquire (epochs/episodes to threshold).

Goal: identify the simplest bio model that still competes with backprop.
"""

import numpy as np
import time
from classification_experiments import (
    BackpropClassifier, BioClassifier, load_digits_data, one_hot,
    tanh_act, tanh_deriv, softmax,
)
from experiments import (
    BackpropRLAgent, BioRLAgent, BackpropNetwork,
    ChainMDP, CartPole, tanh_deriv as td2,
    softmax as sm2, one_hot as oh1, vec_state,
)


# ===================================================================
# Ablated Bio Classifier (from ablation_study.py, extended)
# ===================================================================

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

        if self.use_plateau:
            err_mag = np.mean(np.abs(error))
            surprise = err_mag / (self.error_ema + 1e-6)
            plateau = 1.0 + (self.plateau_gain - 1.0) * min(surprise, 3.0) / 3.0
            self.error_ema = 0.95 * self.error_ema + 0.05 * err_mag
        else:
            plateau = 1.0

        residuals = [None] * self.nl
        residuals[-1] = error / n

        for i in range(self.nl - 2, -1, -1):
            if self.use_sd:
                dendritic_target = error @ self.B[i] / n
            else:
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


# ===================================================================
# Ablated Bio RL Agent
# ===================================================================

class AblatedBioRLAgent(BioRLAgent):
    """BioRLAgent with toggles to disable individual components."""

    def __init__(self, state_dim, hidden, n_actions, lr=0.01, plateau_gain=3.0,
                 trace_decay=0.85, silent_thresh=3.0, ei_tau=0.05,
                 pred_lr=0.01, seed=42,
                 use_sd=True, use_bcp=True, use_btsp=True,
                 use_plateau=True, use_silent=True):
        super().__init__(state_dim, hidden, n_actions, lr=lr,
                         plateau_gain=plateau_gain, trace_decay=trace_decay,
                         silent_thresh=silent_thresh, ei_tau=ei_tau,
                         pred_lr=pred_lr, seed=seed)
        self.use_sd = use_sd
        self.use_bcp = use_bcp
        self.use_btsp = use_btsp
        self.use_plateau = use_plateau
        self.use_silent = use_silent

    def act(self, state, rng, epsilon=0.0):
        z1 = state @ self.W1 + self.b1
        h = tanh_act(z1)
        z2 = h @ self.W2 + self.b2
        probs = softmax(z2)

        if rng.random() < epsilon:
            action = rng.randint(self.n_actions)
        else:
            action = rng.choice(self.n_actions, p=probs.ravel())

        pred_reward = (h @ self.Wr + self.br).item()

        action_onehot = np.zeros_like(probs)
        action_onehot[0, action] = 1.0
        policy_error = action_onehot - probs

        hebbian_W2 = h.T @ policy_error
        hebbian_b2 = policy_error.copy()

        if self.use_sd:
            hidden_signal = policy_error @ self.B * tanh_deriv(z1)
        else:
            hidden_signal = np.zeros_like(z1)

        hebbian_W1 = state.T @ hidden_signal
        hebbian_b1 = hidden_signal.copy()

        if self.use_btsp:
            self.trace_W2 = self.trace_decay * self.trace_W2 + hebbian_W2
            self.trace_b2 = self.trace_decay * self.trace_b2 + hebbian_b2
            self.trace_W1 = self.trace_decay * self.trace_W1 + hebbian_W1
            self.trace_b1 = self.trace_decay * self.trace_b1 + hebbian_b1
        else:
            # No trace accumulation — just immediate Hebbian
            self.trace_W2 = hebbian_W2
            self.trace_b2 = hebbian_b2
            self.trace_W1 = hebbian_W1
            self.trace_b1 = hebbian_b1

        self.last_state = state
        self.last_hidden = h
        self.last_pre_h = z1
        self.last_probs = probs
        self.last_action = action

        return action, pred_reward

    def learn(self, td_target, pred_reward):
        h = self.last_hidden
        z1 = self.last_pre_h
        state = self.last_state

        rpe = td_target - pred_reward
        rpe = np.clip(rpe, -5.0, 5.0)

        # Adaptive plateau
        if self.use_plateau:
            surprise = abs(rpe) / (self.rpe_ema + 1e-6)
            plateau = 1.0 + (self.plateau_gain - 1.0) * min(surprise, 3.0) / 3.0
            self.rpe_ema = 0.95 * self.rpe_ema + 0.05 * abs(rpe)
        else:
            plateau = 1.0

        # SD Residuals
        if self.use_sd:
            action_onehot = np.zeros((1, self.n_actions))
            action_onehot[0, self.last_action] = 1.0
            rpe_signal = rpe * (action_onehot - self.last_probs)
            dendritic_input = rpe_signal @ self.B
        else:
            dendritic_input = np.zeros_like(z1)

        # BCP
        if self.use_bcp:
            ei_error = h - self.I_pred
            self.I_pred += self.ei_tau * (h - self.I_pred)
        else:
            ei_error = 0.0

        sd_hidden = (dendritic_input + 0.1 * ei_error) * tanh_deriv(z1)

        # Silent synapse masks
        if self.use_silent:
            mask2 = self._silent_mask(self.W2)
            mask1 = self._silent_mask(self.W1)
        else:
            mask2 = 1.0
            mask1 = 1.0

        # Three-factor update
        self.W2 += self.lr * plateau * rpe * self.trace_W2 * mask2
        self.b2 += self.lr * plateau * rpe * self.trace_b2

        self.W1 += self.lr * plateau * rpe * self.trace_W1 * mask1
        self.b1 += self.lr * plateau * rpe * self.trace_b1

        # SD residual correction
        dW1_local = state.T @ (sd_hidden * plateau) / state.shape[0]
        self.W1 += 0.2 * self.lr * dW1_local * mask1
        self.b1 += 0.2 * self.lr * np.mean(sd_hidden * plateau, axis=0, keepdims=True)

        # Update reward predictor
        pred_error = np.clip(td_target - (h @ self.Wr + self.br), -5.0, 5.0)
        self.Wr += self.pred_lr * h.T * pred_error
        self.br += self.pred_lr * pred_error

        return rpe


# ===================================================================
# Experiment runners
# ===================================================================

def run_classification(config_name, cfg, epochs=50, batch_size=32):
    """Run classification on digits. Track accuracy every epoch + time-to-threshold."""
    from classification_experiments import load_digits_data, one_hot as oh
    X_tr, y_tr, X_te, y_te, nc = load_digits_data()
    Y_tr = oh(y_tr, nc)
    arch = [64, 128, 64, nc]

    if cfg is None:
        model = BackpropClassifier(arch, lr=0.2, seed=7)
    else:
        model = AblatedBioClassifier(arch, lr=0.3, plateau_gain=3.0,
                                      trace_decay=0.7, seed=7, **cfg)

    rng = np.random.RandomState(42)
    losses = []
    test_accs = []
    first_80 = None
    first_90 = None
    first_95 = None

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X_tr))
        ep_loss = 0; nb = 0
        for s in range(0, len(X_tr), batch_size):
            e = min(s + batch_size, len(X_tr))
            bi = idx[s:e]
            ep_loss += model.train_step(X_tr[bi], Y_tr[bi])
            nb += 1
        losses.append(ep_loss / nb)

        te = model.accuracy(X_te, y_te)
        test_accs.append((ep, te))
        if first_80 is None and te >= 0.80: first_80 = ep
        if first_90 is None and te >= 0.90: first_90 = ep
        if first_95 is None and te >= 0.95: first_95 = ep

    final_acc = test_accs[-1][1]
    return {"losses": losses, "test_accs": test_accs, "final_acc": final_acc,
            "first_80": first_80, "first_90": first_90, "first_95": first_95}


def run_chain_rl(config_name, cfg, length=20, n_ep=600):
    """Run chain MDP. Track per-episode rewards + time-to-first-solve + sustained solve."""
    if cfg is None:
        agent = BackpropRLAgent(length, 32, 2, lr=0.01, pred_lr=0.02,
                                 trace_decay=0.95, entropy_coeff=0.02, seed=7)
    else:
        agent = AblatedBioRLAgent(length, 32, 2, lr=0.01, plateau_gain=3.0,
                                   trace_decay=0.95, pred_lr=0.02, seed=7, **cfg)

    env = ChainMDP(length=length, slip=0.1, seed=0)
    rng = np.random.RandomState(42)
    ep_rewards = []
    first_solve = None       # first episode with reward > 0
    first_consistent = None  # first episode where rolling 20-ep avg > 0.5

    for ep in range(1, n_ep + 1):
        s = env.reset()
        agent.reset_traces()
        eps = max(0.05, 0.3 - 0.25 * ep / n_ep)
        total_r = 0

        for t in range(length * 5):
            sv = oh1(s, length)
            action, pred_r = agent.act(sv, rng, epsilon=eps)
            s2, r, done = env.step(action)
            total_r += r

            if done:
                td_target = r
            else:
                next_val = agent.predict_value(oh1(s2, length))
                td_target = r + 0.99 * next_val

            agent.learn(td_target, pred_r)
            s = s2
            if done:
                break

        ep_rewards.append(total_r)
        if first_solve is None and total_r > 0:
            first_solve = ep
        if first_consistent is None and ep >= 20:
            if np.mean(ep_rewards[-20:]) > 0.5:
                first_consistent = ep

    final_avg = np.mean(ep_rewards[-50:])
    return {"ep_rewards": ep_rewards, "final_avg": final_avg,
            "first_solve": first_solve, "first_consistent": first_consistent}


def run_cartpole_rl(config_name, cfg, n_ep=2000):
    """Run cartpole with tuned hyperparameters. Track steps + time-to-threshold."""
    cp_scale = np.array([2.4, 4.0, 0.21, 4.0])

    if cfg is None:
        agent = BackpropRLAgent(4, 64, 2, lr=0.005, pred_lr=0.05,
                                 trace_decay=0.9, entropy_coeff=0.01, seed=7)
    else:
        agent = AblatedBioRLAgent(4, 64, 2, lr=0.005, plateau_gain=3.0,
                                   trace_decay=0.9, pred_lr=0.05, seed=7, **cfg)

    env = CartPole(seed=0)
    rng = np.random.RandomState(42)
    ep_steps = []
    first_50 = None    # first episode where rolling avg > 50 steps
    first_100 = None   # first episode where rolling avg > 100 steps
    first_150 = None   # first episode where rolling avg > 150 steps

    for ep in range(1, n_ep + 1):
        s = env.reset()
        agent.reset_traces()
        eps = max(0.01, 0.2 - 0.19 * ep / n_ep)
        steps = 0

        for t in range(500):
            sv = vec_state(s) / cp_scale
            action, pred_r = agent.act(sv, rng, epsilon=eps)
            s2, r, done = env.step(action)
            steps += 1

            if done:
                td_target = -1.0  # penalty for falling
            else:
                next_val = agent.predict_value(vec_state(s2) / cp_scale)
                td_target = r + 0.99 * next_val

            agent.learn(td_target, pred_r)
            s = s2
            if done:
                break

        ep_steps.append(steps)
        if ep >= 20:
            avg20 = np.mean(ep_steps[-20:])
            if first_50 is None and avg20 >= 50: first_50 = ep
            if first_100 is None and avg20 >= 100: first_100 = ep
            if first_150 is None and avg20 >= 150: first_150 = ep

    final_avg = np.mean(ep_steps[-50:])
    return {"ep_steps": ep_steps, "final_avg": final_avg,
            "first_50": first_50, "first_100": first_100, "first_150": first_150}


# ===================================================================
# Main experiment
# ===================================================================

def main():
    # --- Define all configurations ---
    # Full bio and backprop baselines
    configs = {
        "Backprop":       None,
        "Bio (full)":     dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=True),
    }

    # Single ablations (remove one)
    single_ablations = {
        "- SD":           dict(use_sd=False, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=True),
        "- BCP":          dict(use_sd=True, use_bcp=False, use_btsp=True, use_plateau=True, use_silent=True),
        "- BTSP":         dict(use_sd=True, use_bcp=True, use_btsp=False, use_plateau=True, use_silent=True),
        "- Plateau":      dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=False, use_silent=True),
        "- Silent":       dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=False),
    }

    # Minimal combos (keep only named components + SD which is essential)
    minimal_combos = {
        "SD only":        dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=False, use_silent=False),
        "SD+Plateau":     dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=True, use_silent=False),
        "SD+BTSP":        dict(use_sd=True, use_bcp=False, use_btsp=True, use_plateau=False, use_silent=False),
        "SD+BTSP+Plat":   dict(use_sd=True, use_bcp=False, use_btsp=True, use_plateau=True, use_silent=False),
        "SD+Silent":      dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=False, use_silent=True),
        "SD+Plat+Sil":    dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=True, use_silent=True),
    }

    all_configs = {**configs, **single_ablations, **minimal_combos}

    # --- Run all experiments ---
    results = {}
    print("=" * 70)
    print("PART 1: CLASSIFICATION (sklearn digits, 50 epochs)")
    print("=" * 70)

    for name, cfg in all_configs.items():
        t0 = time.time()
        r = run_classification(name, cfg)
        dt = time.time() - t0
        results.setdefault(name, {})["classification"] = r
        f80 = str(r['first_80']) if r['first_80'] else "—"
        f90 = str(r['first_90']) if r['first_90'] else "—"
        f95 = str(r['first_95']) if r['first_95'] else "—"
        print(f"  {name:>18s}: acc={r['final_acc']:.4f}  "
              f"@80%={f80:>3s}  @90%={f90:>3s}  @95%={f95:>3s}  ({dt:.1f}s)")

    print()
    print("=" * 70)
    print("PART 2: CHAIN MDP (length=20, 600 episodes)")
    print("=" * 70)

    for name, cfg in all_configs.items():
        t0 = time.time()
        r = run_chain_rl(name, cfg)
        dt = time.time() - t0
        results[name]["chain"] = r
        fs = str(r['first_solve']) if r['first_solve'] else "—"
        fc = str(r['first_consistent']) if r['first_consistent'] else "—"
        print(f"  {name:>18s}: avg={r['final_avg']:.3f}  "
              f"1st_solve={fs:>4s}  consistent={fc:>4s}  ({dt:.1f}s)")

    print()
    print("=" * 70)
    print("PART 3: CARTPOLE (2000 episodes, tuned)")
    print("=" * 70)

    for name, cfg in all_configs.items():
        t0 = time.time()
        r = run_cartpole_rl(name, cfg)
        dt = time.time() - t0
        results[name]["cartpole"] = r
        f50 = str(r['first_50']) if r['first_50'] else "—"
        f100 = str(r['first_100']) if r['first_100'] else "—"
        f150 = str(r['first_150']) if r['first_150'] else "—"
        print(f"  {name:>18s}: avg={r['final_avg']:.1f}  "
              f"@50={f50:>5s}  @100={f100:>5s}  @150={f150:>5s}  ({dt:.1f}s)")

    # --- Summary table with acquisition speed ---
    print()
    print("=" * 80)
    print("FULL SUMMARY: FINAL PERFORMANCE + ACQUISITION SPEED")
    print("=" * 80)
    header = (f"  {'Config':>18s}  {'#':>2s}  "
              f"{'Acc':>5s} {'@90':>4s} {'@95':>4s}  "
              f"{'ChR':>5s} {'1stS':>4s} {'Cons':>4s}  "
              f"{'CPst':>5s} {'@50':>5s} {'@100':>5s}")
    print(header)
    print("  " + "-" * 76)

    for name in all_configs:
        r = results[name]
        cfg = all_configs[name]
        nc = "BP" if cfg is None else str(sum(cfg.values()))

        cr = r["classification"]
        ch = r["chain"]
        cp = r["cartpole"]

        def fmt(v, none_str="—"):
            return none_str if v is None else str(v)

        print(f"  {name:>18s}  {nc:>2s}  "
              f"{cr['final_acc']:5.3f} {fmt(cr['first_90']):>4s} {fmt(cr['first_95']):>4s}  "
              f"{ch['final_avg']:5.3f} {fmt(ch['first_solve']):>4s} {fmt(ch['first_consistent']):>4s}  "
              f"{cp['final_avg']:5.1f} {fmt(cp['first_50']):>5s} {fmt(cp['first_100']):>5s}")

    # --- Plot ---
    plot_results(results, all_configs)

    return results


def plot_results(results, all_configs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    key_configs = ["Backprop", "Bio (full)", "SD only", "SD+Plateau",
                   "SD+BTSP", "SD+BTSP+Plat"]
    colors = {
        "Backprop": "#1f77b4", "Bio (full)": "#d62728",
        "SD only": "#ff7f0e", "SD+Plateau": "#2ca02c",
        "SD+BTSP": "#9467bd", "SD+BTSP+Plat": "#e377c2",
        "- SD": "#888888", "- BCP": "#17becf", "- BTSP": "#bcbd22",
        "- Plateau": "#8c564b", "- Silent": "#e377c2",
        "SD+Silent": "#7f7f7f", "SD+Plat+Sil": "#aec7e8",
    }

    # === Figure 1: Learning curves (3 tasks) ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Classification
    ax = axes[0]
    for name in key_configs:
        r = results[name]["classification"]
        eps, accs = zip(*r["test_accs"])
        ax.plot(eps, accs, label=name, color=colors.get(name, "gray"),
                alpha=0.85, marker='o', markersize=2)
    ax.axhline(y=0.90, color='gray', ls=':', alpha=0.4)
    ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)
    ax.set_title("Classification (Digits) — Test Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.05]); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Chain
    ax = axes[1]
    w = 20
    for name in key_configs:
        r = results[name]["chain"]
        vals = np.array(r["ep_rewards"])
        smoothed = np.convolve(vals, np.ones(w)/w, 'valid')
        ax.plot(smoothed, label=name, color=colors.get(name, "gray"), alpha=0.85)
    ax.set_title("Chain MDP — Reward (rolling 20)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # CartPole
    ax = axes[2]
    w = 30
    for name in key_configs:
        r = results[name]["cartpole"]
        vals = np.array(r["ep_steps"])
        smoothed = np.convolve(vals, np.ones(w)/w, 'valid')
        ax.plot(smoothed, label=name, color=colors.get(name, "gray"), alpha=0.85)
    ax.axhline(y=50, color='gray', ls=':', alpha=0.4)
    ax.axhline(y=100, color='gray', ls=':', alpha=0.4)
    ax.set_title("CartPole — Steps (rolling 30)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("/Users/sbae703/Research/ML/local_max/simplicity_curves.png", dpi=150)
    print("  Saved simplicity_curves.png")

    # === Figure 2: Acquisition speed comparison ===
    all_names = list(all_configs.keys())

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

    def color_for(name):
        cfg = all_configs[name]
        if cfg is None: return '#1f77b4'
        nc = sum(cfg.values())
        if nc == 5: return '#d62728'
        if nc <= 2: return '#2ca02c'
        return '#9467bd'

    # --- Top row: final performance ---
    # Classification final accuracy
    ax = axes2[0, 0]
    vals = [results[n]["classification"]["final_acc"] for n in all_names]
    cs = [color_for(n) for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Test Accuracy"); ax.set_title("Classification — Final Accuracy")
    ax.set_xlim([0, 1.05]); ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, vals):
        ax.text(min(val + 0.005, 1.0), bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', fontsize=7)

    # Chain final reward
    ax = axes2[0, 1]
    vals = [results[n]["chain"]["final_avg"] for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Avg Reward (last 50)"); ax.set_title("Chain — Final Reward")
    ax.grid(True, alpha=0.3, axis='x')

    # CartPole final steps
    ax = axes2[0, 2]
    vals = [results[n]["cartpole"]["final_avg"] for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Avg Steps (last 50)"); ax.set_title("CartPole — Final Steps")
    ax.grid(True, alpha=0.3, axis='x')

    # --- Bottom row: acquisition speed ---
    max_ep_class = 50
    max_ep_chain = 600
    max_ep_cart = 2000

    # Classification: epochs to 90%
    ax = axes2[1, 0]
    vals = [results[n]["classification"]["first_90"] or max_ep_class for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Epochs to 90% accuracy"); ax.set_title("Classification — Acquisition Speed")
    ax.set_xlim([0, max_ep_class + 5]); ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()  # fewer epochs = better = longer bar toward right
    for bar, val, name in zip(bars, vals, all_names):
        r90 = results[name]["classification"]["first_90"]
        label = str(val) if r90 else "never"
        ax.text(val - 0.5, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontsize=7, color='white', fontweight='bold')

    # Chain: episodes to consistent solve
    ax = axes2[1, 1]
    vals = [results[n]["chain"]["first_consistent"] or max_ep_chain for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Episodes to consistent solve (avg>0.5)"); ax.set_title("Chain — Acquisition Speed")
    ax.set_xlim([0, max_ep_chain + 20]); ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()
    for bar, val, name in zip(bars, vals, all_names):
        fc = results[name]["chain"]["first_consistent"]
        label = str(val) if fc else "never"
        ax.text(val - 2, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontsize=7, color='white', fontweight='bold')

    # CartPole: episodes to 50 avg steps
    ax = axes2[1, 2]
    vals = [results[n]["cartpole"]["first_50"] or max_ep_cart for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Episodes to avg>50 steps"); ax.set_title("CartPole — Acquisition Speed")
    ax.set_xlim([0, max_ep_cart + 50]); ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()
    for bar, val, name in zip(bars, vals, all_names):
        f50 = results[name]["cartpole"]["first_50"]
        label = str(val) if f50 else "never"
        ax.text(val - 5, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontsize=7, color='white', fontweight='bold')

    fig2.tight_layout()
    fig2.savefig("/Users/sbae703/Research/ML/local_max/simplicity_full.png", dpi=150)
    print("  Saved simplicity_full.png")


if __name__ == "__main__":
    main()
