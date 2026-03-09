"""
Focused experiment: proper backprop baseline vs interesting bio configs.

Backprop now uses TD(λ) eligibility traces, entropy bonus, advantage
normalization, and full value gradient — a fair comparison.

Bio configs selected from simplicity_full.png results:
  - SD only:      minimal bio baseline (SD residuals alone)
  - SD+Plateau:   fast acquisition, simple
  - SD+Silent:    strong on CartPole
  - SD+Plat+Sil:  best minimal combo overall
  - Bio (full):   reference (all 5 components)
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
from simplicity_experiment import AblatedBioClassifier, AblatedBioRLAgent


# ===================================================================
# Configs
# ===================================================================

CONFIGS = {
    "Backprop":     None,
    "Bio (full)":   dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=True),
    "SD only":      dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=False, use_silent=False),
    "SD+Plateau":   dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=True, use_silent=False),
    "SD+Silent":    dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=False, use_silent=True),
    "SD+Plat+Sil":  dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=True, use_silent=True),
}


# ===================================================================
# Experiment runners
# ===================================================================

def run_classification(cfg, epochs=50, batch_size=32):
    X_tr, y_tr, X_te, y_te, nc = load_digits_data()
    Y_tr = one_hot(y_tr, nc)
    arch = [64, 128, 64, nc]

    if cfg is None:
        model = BackpropClassifier(arch, lr=0.2, seed=7)
    else:
        model = AblatedBioClassifier(arch, lr=0.3, plateau_gain=3.0,
                                      trace_decay=0.7, seed=7, **cfg)

    rng = np.random.RandomState(42)
    test_accs = []
    first_90 = None
    first_95 = None

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X_tr))
        for s in range(0, len(X_tr), batch_size):
            e = min(s + batch_size, len(X_tr))
            model.train_step(X_tr[idx[s:e]], Y_tr[idx[s:e]])

        te = model.accuracy(X_te, y_te)
        test_accs.append((ep, te))
        if first_90 is None and te >= 0.90: first_90 = ep
        if first_95 is None and te >= 0.95: first_95 = ep

    return {"test_accs": test_accs, "final_acc": test_accs[-1][1],
            "first_90": first_90, "first_95": first_95}


def run_chain(cfg, length=20, n_ep=600):
    if cfg is None:
        agent = BackpropRLAgent(length, 32, 2, lr=0.01, pred_lr=0.02,
                                 trace_decay=0.95, entropy_coeff=0.02, seed=7)
    else:
        agent = AblatedBioRLAgent(length, 32, 2, lr=0.01, plateau_gain=3.0,
                                   trace_decay=0.95, pred_lr=0.02, seed=7, **cfg)

    env = ChainMDP(length=length, slip=0.1, seed=0)
    rng = np.random.RandomState(42)
    ep_rewards = []
    first_solve = None
    first_consistent = None

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
        # Early stop: if rolling 50-ep avg > 0.9, task is solved
        if ep >= 100 and np.mean(ep_rewards[-50:]) > 0.9:
            break

    return {"ep_rewards": ep_rewards, "final_avg": np.mean(ep_rewards[-50:]),
            "first_solve": first_solve, "first_consistent": first_consistent}


def run_cartpole(cfg, n_ep=2000):
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
    first_50 = None
    first_100 = None

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
                td_target = -1.0
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
        # Early stop: if rolling 50-ep avg > 400, task is solved
        if ep >= 100 and np.mean(ep_steps[-50:]) > 400:
            break

    return {"ep_steps": ep_steps, "final_avg": np.mean(ep_steps[-50:]),
            "first_50": first_50, "first_100": first_100}


# ===================================================================
# Plotting
# ===================================================================

def plot_results(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(CONFIGS.keys())
    colors = {
        "Backprop": "#1f77b4", "Bio (full)": "#d62728",
        "SD only": "#ff7f0e", "SD+Plateau": "#2ca02c",
        "SD+Silent": "#7f7f7f", "SD+Plat+Sil": "#9467bd",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Top row: learning curves ---
    # Classification
    ax = axes[0, 0]
    for name in names:
        r = results[name]["classification"]
        eps, accs = zip(*r["test_accs"])
        ax.plot(eps, accs, label=name, color=colors.get(name, "gray"), alpha=0.85)
    ax.axhline(y=0.90, color='gray', ls=':', alpha=0.4)
    ax.set_title("Classification — Test Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.05]); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Chain
    ax = axes[0, 1]
    w = 20
    for name in names:
        r = results[name]["chain"]
        vals = np.array(r["ep_rewards"])
        smoothed = np.convolve(vals, np.ones(w)/w, 'valid')
        ax.plot(smoothed, label=name, color=colors.get(name, "gray"), alpha=0.85)
    ax.set_title("Chain MDP — Reward (rolling 20)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # CartPole
    ax = axes[0, 2]
    w = 30
    for name in names:
        r = results[name]["cartpole"]
        vals = np.array(r["ep_steps"])
        smoothed = np.convolve(vals, np.ones(w)/w, 'valid')
        ax.plot(smoothed, label=name, color=colors.get(name, "gray"), alpha=0.85)
    ax.axhline(y=100, color='gray', ls=':', alpha=0.4)
    ax.set_title("CartPole — Steps (rolling 30)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Bottom row: bar charts (final perf + acquisition speed) ---
    def color_for(name):
        return colors.get(name, "gray")

    # Classification final + speed
    ax = axes[1, 0]
    vals = [results[n]["classification"]["final_acc"] for n in names]
    cs = [color_for(n) for n in names]
    bars = ax.barh(range(len(names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Test Accuracy"); ax.set_title("Classification — Final")
    ax.set_xlim([0, 1.05]); ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, vals):
        ax.text(min(val + 0.005, 1.0), bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', fontsize=8)

    # Chain final + speed
    ax = axes[1, 1]
    vals = [results[n]["chain"]["first_consistent"] or 600 for n in names]
    bars = ax.barh(range(len(names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Episodes to consistent (fewer=better)")
    ax.set_title("Chain — Acquisition Speed")
    ax.set_xlim([0, 620]); ax.grid(True, alpha=0.3, axis='x'); ax.invert_xaxis()
    for bar, val, name in zip(bars, vals, names):
        fc = results[name]["chain"]["first_consistent"]
        label = str(val) if fc else "never"
        ax.text(val - 2, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontsize=8, color='white', fontweight='bold')

    # CartPole speed
    ax = axes[1, 2]
    vals = [results[n]["cartpole"]["first_50"] or 2000 for n in names]
    bars = ax.barh(range(len(names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Episodes to avg>50 (fewer=better)")
    ax.set_title("CartPole — Acquisition Speed")
    ax.set_xlim([0, 2050]); ax.grid(True, alpha=0.3, axis='x'); ax.invert_xaxis()
    for bar, val, name in zip(bars, vals, names):
        f50 = results[name]["cartpole"]["first_50"]
        label = str(val) if f50 else "never"
        ax.text(val - 5, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontsize=8, color='white', fontweight='bold')

    fig.tight_layout()
    fig.savefig("focused_results.png", dpi=150)
    print("  Saved focused_results.png")


# ===================================================================
# Main
# ===================================================================

def main():
    results = {}

    for task_name, runner in [("classification", run_classification),
                               ("chain", run_chain),
                               ("cartpole", run_cartpole)]:
        print("=" * 70)
        print(f"  {task_name.upper()}")
        print("=" * 70)

        for name, cfg in CONFIGS.items():
            t0 = time.time()
            r = runner(cfg)
            dt = time.time() - t0
            results.setdefault(name, {})[task_name] = r

            if task_name == "classification":
                print(f"  {name:>14s}: acc={r['final_acc']:.4f}  "
                      f"@90%={str(r['first_90']) if r['first_90'] else '—':>3s}  "
                      f"@95%={str(r['first_95']) if r['first_95'] else '—':>3s}  ({dt:.1f}s)")
            elif task_name == "chain":
                fs = str(r['first_solve']) if r['first_solve'] else "—"
                fc = str(r['first_consistent']) if r['first_consistent'] else "—"
                print(f"  {name:>14s}: avg={r['final_avg']:.3f}  "
                      f"1st={fs:>4s}  cons={fc:>4s}  ({dt:.1f}s)")
            else:
                f50 = str(r['first_50']) if r['first_50'] else "—"
                f100 = str(r['first_100']) if r['first_100'] else "—"
                print(f"  {name:>14s}: avg={r['final_avg']:.1f}  "
                      f"@50={f50:>5s}  @100={f100:>5s}  ({dt:.1f}s)")

    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Config':>14s}  {'Acc':>5s} {'@90':>4s}  "
          f"{'ChR':>5s} {'Cons':>4s}  {'CPst':>5s} {'@50':>5s}")
    print("  " + "-" * 55)
    for name in CONFIGS:
        r = results[name]
        cr = r["classification"]; ch = r["chain"]; cp = r["cartpole"]
        def fmt(v): return "—" if v is None else str(v)
        print(f"  {name:>14s}  {cr['final_acc']:5.3f} {fmt(cr['first_90']):>4s}  "
              f"{ch['final_avg']:5.3f} {fmt(ch['first_consistent']):>4s}  "
              f"{cp['final_avg']:5.1f} {fmt(cp['first_50']):>5s}")

    plot_results(results)
    return results


if __name__ == "__main__":
    main()
