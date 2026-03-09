"""
Simplicity experiment (GPU / PyTorch version)
Find the minimal set of bio components across classification + temporal tasks.

Metrics: final performance AND time-to-acquire (epochs/episodes to threshold).

Self-contained — no imports from other local files.
Run: python simplicity_gpu.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ===================================================================
# Activations (torch)
# ===================================================================

def tanh_deriv(x):
    return 1.0 - torch.tanh(x) ** 2


def softmax(x):
    return F.softmax(x, dim=-1)


# ===================================================================
# Data loading
# ===================================================================

def load_digits_data():
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data / 16.0
    y = d.target.astype(int)
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    X_tr = torch.tensor(X[idx[:split]], dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y[idx[:split]], dtype=torch.long, device=DEVICE)
    X_te = torch.tensor(X[idx[split:]], dtype=torch.float32, device=DEVICE)
    y_te = torch.tensor(y[idx[split:]], dtype=torch.long, device=DEVICE)
    return X_tr, y_tr, X_te, y_te, 10


def one_hot(y, k):
    return F.one_hot(y, k).float()


# ===================================================================
# Backprop Classifier (standard MLP, autograd)
# ===================================================================

class BackpropClassifier:
    def __init__(self, sizes, lr=0.1, seed=42):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.lr = lr
        self.nl = len(sizes) - 1
        self.W = []
        self.b = []
        for i in range(self.nl):
            w = torch.randn(sizes[i], sizes[i+1], generator=g) * np.sqrt(2.0/sizes[i])
            self.W.append(w.to(DEVICE))
            self.b.append(torch.zeros(1, sizes[i+1], device=DEVICE))

    def forward(self, x):
        a = [x]; z = [x]
        for i in range(self.nl):
            pre = a[-1] @ self.W[i] + self.b[i]
            z.append(pre)
            if i < self.nl - 1:
                a.append(torch.tanh(pre))
            else:
                a.append(softmax(pre))
        return a, z

    def train_step(self, x, y_oh):
        a, z = self.forward(x)
        probs = a[-1]
        n = x.shape[0]
        loss = -torch.mean(torch.sum(y_oh * torch.log(probs + 1e-12), dim=1))

        d = (probs - y_oh) / n
        for i in reversed(range(self.nl)):
            self.W[i] -= self.lr * (a[i].T @ d)
            self.b[i] -= self.lr * torch.sum(d, dim=0, keepdim=True)
            if i > 0:
                d = (d @ self.W[i].T) * tanh_deriv(z[i])
        return loss.item()

    def accuracy(self, x, y):
        with torch.no_grad():
            pred = torch.argmax(self.forward(x)[0][-1], dim=1)
            return (pred == y).float().mean().item()


# ===================================================================
# Bio Classifier with ablation toggles
# ===================================================================

class BioClassifier:
    def __init__(self, sizes, lr=0.1, plateau_gain=3.0, trace_decay=0.7,
                 silent_thresh=3.0, ei_tau=0.05, seed=42,
                 use_sd=True, use_bcp=True, use_btsp=True,
                 use_plateau=True, use_silent=True):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.lr = lr
        self.plateau_gain = plateau_gain
        self.trace_decay = trace_decay
        self.silent_thresh = silent_thresh
        self.ei_tau = ei_tau
        self.nl = len(sizes) - 1
        self.sizes = sizes
        self.use_sd = use_sd
        self.use_bcp = use_bcp
        self.use_btsp = use_btsp
        self.use_plateau = use_plateau
        self.use_silent = use_silent

        self.W = []
        self.b = []
        for i in range(self.nl):
            w = torch.randn(sizes[i], sizes[i+1], generator=g) * np.sqrt(2.0/sizes[i])
            self.W.append(w.to(DEVICE))
            self.b.append(torch.zeros(1, sizes[i+1], device=DEVICE))

        out = sizes[-1]
        self.B = [
            (torch.randn(out, sizes[i+1], generator=g) * np.sqrt(1.0/out)).to(DEVICE)
            for i in range(self.nl - 1)
        ]

        self.I_pred = [torch.zeros(1, sizes[i+1], device=DEVICE) for i in range(self.nl)]
        self.traces = [torch.zeros_like(w) for w in self.W]
        self.traces_b = [torch.zeros_like(b) for b in self.b]
        self.error_ema = 0.5

    def _silent_mask(self, W):
        return 1.0 / (1.0 + torch.abs(W) / self.silent_thresh)

    def forward(self, x):
        a = [x]; z = [x]
        for i in range(self.nl):
            pre = a[-1] @ self.W[i] + self.b[i]
            z.append(pre)
            if i < self.nl - 1:
                a.append(torch.tanh(pre))
            else:
                a.append(softmax(pre))
        return a, z

    def train_step(self, x, y_oh):
        a, z = self.forward(x)
        probs = a[-1]
        n = x.shape[0]

        loss = -torch.mean(torch.sum(y_oh * torch.log(probs + 1e-12), dim=1))
        error = y_oh - probs

        if self.use_plateau:
            err_mag = torch.mean(torch.abs(error)).item()
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
                dendritic_target = torch.zeros(n, self.sizes[i + 1], device=DEVICE)

            if self.use_bcp:
                ei_balance = a[i + 1] - self.I_pred[i]
                self.I_pred[i] = self.I_pred[i] + self.ei_tau * torch.mean(
                    a[i + 1] - self.I_pred[i], dim=0, keepdim=True)
                bcp_term = 0.01 * ei_balance
            else:
                bcp_term = 0.0

            sd = (dendritic_target + bcp_term) * tanh_deriv(z[i + 1])
            residuals[i] = sd

        for i in range(self.nl):
            local_err = residuals[i] * plateau
            dW = a[i].T @ local_err
            db = torch.sum(local_err, dim=0, keepdim=True)

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

            self.W[i] = self.W[i] + self.lr * (dW + trace_contrib) * mask
            self.b[i] = self.b[i] + self.lr * (db + trace_contrib_b)
            self.W[i] = torch.clamp(self.W[i], -5.0, 5.0)

        return loss.item()

    def accuracy(self, x, y):
        with torch.no_grad():
            pred = torch.argmax(self.forward(x)[0][-1], dim=1)
            return (pred == y).float().mean().item()


# ===================================================================
# Environments (stay on CPU — sequential by nature)
# ===================================================================

class ChainMDP:
    def __init__(self, length=10, slip=0.1, seed=42):
        self.L = length; self.slip = slip
        self.rng = np.random.RandomState(seed); self.s = 0

    def reset(self):
        self.s = 0; return self.s

    def step(self, a):
        if self.rng.random() < self.slip: a = 1 - a
        self.s = max(0, min(self.L - 1, self.s + (1 if a else -1)))
        d = self.s == self.L - 1
        return self.s, (1.0 if d else 0.0), d


class CartPole:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed); self.state = None; self.reset()

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4); return self.state.copy()

    def step(self, a):
        x, xd, th, thd = self.state
        f = 10.0 if a else -10.0; c, s = np.cos(th), np.sin(th)
        tmp = (f + 0.05 * thd**2 * s) / 1.1
        tha = (9.8 * s - c * tmp) / (0.5 * (4/3 - 0.1 * c**2 / 1.1))
        xa = tmp - 0.05 * tha * c / 1.1
        self.state = np.array([x + 0.02*xd, xd + 0.02*xa, th + 0.02*thd, thd + 0.02*tha])
        d = abs(self.state[0]) > 2.4 or abs(self.state[2]) > 12 * np.pi / 180
        return self.state.copy(), (0.0 if d else 1.0), d


# ===================================================================
# RL Agents (GPU tensors for weights, CPU scalars for env interaction)
# ===================================================================

class BackpropRLAgent:
    def __init__(self, state_dim, hidden, n_actions, lr=0.01, pred_lr=0.01, seed=42):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.lr = lr; self.pred_lr = pred_lr; self.n_actions = n_actions

        self.W1 = (torch.randn(state_dim, hidden, generator=g) * np.sqrt(2.0/state_dim)).to(DEVICE)
        self.b1 = torch.zeros(1, hidden, device=DEVICE)
        self.W2 = (torch.randn(hidden, n_actions, generator=g) * np.sqrt(2.0/hidden)).to(DEVICE)
        self.b2 = torch.zeros(1, n_actions, device=DEVICE)
        self.Wr = (torch.randn(hidden, 1, generator=g) * np.sqrt(1.0/hidden)).to(DEVICE)
        self.br = torch.zeros(1, 1, device=DEVICE)
        self._last_grad = None

    def predict_value(self, state_t):
        z1 = state_t @ self.W1 + self.b1
        h = torch.tanh(z1)
        return (h @ self.Wr + self.br).item()

    def act(self, state_t, rng, epsilon=0.0):
        z1 = state_t @ self.W1 + self.b1
        h = torch.tanh(z1)
        z2 = h @ self.W2 + self.b2
        probs = softmax(z2)
        pred_r = (h @ self.Wr + self.br).item()

        probs_np = probs.detach().cpu().numpy().ravel()
        if rng.random() < epsilon:
            action = rng.randint(self.n_actions)
        else:
            action = rng.choice(self.n_actions, p=probs_np)

        oh = torch.zeros_like(probs); oh[0, action] = 1.0
        pe = oh - probs
        self._last_grad = (
            state_t.T @ (pe @ self.W2.T * tanh_deriv(z1)),
            pe @ self.W2.T * tanh_deriv(z1),
            h.T @ pe, pe.clone(), h)
        return action, pred_r

    def learn(self, td_target, pred_reward):
        rpe = max(-2.0, min(2.0, td_target - pred_reward))
        if self._last_grad is not None:
            dW1, db1, dW2, db2, h = self._last_grad
            self.W2 = self.W2 + self.lr * rpe * dW2
            self.b2 = self.b2 + self.lr * rpe * db2
            self.W1 = self.W1 + self.lr * rpe * dW1
            self.b1 = self.b1 + self.lr * rpe * db1
            pred_error = max(-2.0, min(2.0, td_target - (h @ self.Wr + self.br).item()))
            self.Wr = self.Wr + self.pred_lr * h.T * pred_error
            self.br = self.br + self.pred_lr * pred_error
        return rpe

    def reset_traces(self):
        pass


class BioRLAgent:
    def __init__(self, state_dim, hidden, n_actions, lr=0.01, plateau_gain=3.0,
                 trace_decay=0.85, silent_thresh=3.0, ei_tau=0.05,
                 pred_lr=0.01, seed=42,
                 use_sd=True, use_bcp=True, use_btsp=True,
                 use_plateau=True, use_silent=True):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.lr = lr
        self.plateau_gain = plateau_gain
        self.trace_decay = trace_decay
        self.silent_thresh = silent_thresh
        self.ei_tau = ei_tau
        self.pred_lr = pred_lr
        self.n_actions = n_actions
        self.use_sd = use_sd
        self.use_bcp = use_bcp
        self.use_btsp = use_btsp
        self.use_plateau = use_plateau
        self.use_silent = use_silent

        self.W1 = (torch.randn(state_dim, hidden, generator=g) * np.sqrt(2.0/state_dim)).to(DEVICE)
        self.b1 = torch.zeros(1, hidden, device=DEVICE)
        self.W2 = (torch.randn(hidden, n_actions, generator=g) * np.sqrt(2.0/hidden)).to(DEVICE)
        self.b2 = torch.zeros(1, n_actions, device=DEVICE)
        self.Wr = (torch.randn(hidden, 1, generator=g) * np.sqrt(1.0/hidden)).to(DEVICE)
        self.br = torch.zeros(1, 1, device=DEVICE)

        self.B = (torch.randn(n_actions, hidden, generator=g) * np.sqrt(1.0/n_actions)).to(DEVICE)
        self.I_pred = torch.zeros(1, hidden, device=DEVICE)

        self.trace_W1 = torch.zeros_like(self.W1)
        self.trace_b1 = torch.zeros_like(self.b1)
        self.trace_W2 = torch.zeros_like(self.W2)
        self.trace_b2 = torch.zeros_like(self.b2)

        self.rpe_ema = 0.1
        self.last_state = None
        self.last_hidden = None
        self.last_pre_h = None
        self.last_probs = None
        self.last_action = None

    def _silent_mask(self, W):
        return 1.0 / (1.0 + torch.abs(W) / self.silent_thresh)

    def predict_value(self, state_t):
        z1 = state_t @ self.W1 + self.b1
        h = torch.tanh(z1)
        return (h @ self.Wr + self.br).item()

    def act(self, state_t, rng, epsilon=0.0):
        z1 = state_t @ self.W1 + self.b1
        h = torch.tanh(z1)
        z2 = h @ self.W2 + self.b2
        probs = softmax(z2)

        probs_np = probs.detach().cpu().numpy().ravel()
        if rng.random() < epsilon:
            action = rng.randint(self.n_actions)
        else:
            action = rng.choice(self.n_actions, p=probs_np)

        pred_reward = (h @ self.Wr + self.br).item()

        action_onehot = torch.zeros_like(probs)
        action_onehot[0, action] = 1.0
        policy_error = action_onehot - probs

        hebbian_W2 = h.T @ policy_error
        hebbian_b2 = policy_error.clone()

        if self.use_sd:
            hidden_signal = policy_error @ self.B * tanh_deriv(z1)
        else:
            hidden_signal = torch.zeros_like(z1)

        hebbian_W1 = state_t.T @ hidden_signal
        hebbian_b1 = hidden_signal.clone()

        if self.use_btsp:
            self.trace_W2 = self.trace_decay * self.trace_W2 + hebbian_W2
            self.trace_b2 = self.trace_decay * self.trace_b2 + hebbian_b2
            self.trace_W1 = self.trace_decay * self.trace_W1 + hebbian_W1
            self.trace_b1 = self.trace_decay * self.trace_b1 + hebbian_b1
        else:
            self.trace_W2 = hebbian_W2
            self.trace_b2 = hebbian_b2
            self.trace_W1 = hebbian_W1
            self.trace_b1 = hebbian_b1

        self.last_state = state_t
        self.last_hidden = h
        self.last_pre_h = z1
        self.last_probs = probs
        self.last_action = action

        return action, pred_reward

    def learn(self, td_target, pred_reward):
        h = self.last_hidden
        z1 = self.last_pre_h
        state = self.last_state

        rpe = max(-5.0, min(5.0, td_target - pred_reward))

        if self.use_plateau:
            surprise = abs(rpe) / (self.rpe_ema + 1e-6)
            plateau = 1.0 + (self.plateau_gain - 1.0) * min(surprise, 3.0) / 3.0
            self.rpe_ema = 0.95 * self.rpe_ema + 0.05 * abs(rpe)
        else:
            plateau = 1.0

        if self.use_sd:
            action_onehot = torch.zeros(1, self.n_actions, device=DEVICE)
            action_onehot[0, self.last_action] = 1.0
            rpe_signal = rpe * (action_onehot - self.last_probs)
            dendritic_input = rpe_signal @ self.B
        else:
            dendritic_input = torch.zeros_like(z1)

        if self.use_bcp:
            ei_error = h - self.I_pred
            self.I_pred = self.I_pred + self.ei_tau * (h - self.I_pred)
        else:
            ei_error = 0.0

        sd_hidden = (dendritic_input + 0.1 * ei_error) * tanh_deriv(z1)

        if self.use_silent:
            mask2 = self._silent_mask(self.W2)
            mask1 = self._silent_mask(self.W1)
        else:
            mask2 = 1.0
            mask1 = 1.0

        self.W2 = self.W2 + self.lr * plateau * rpe * self.trace_W2 * mask2
        self.b2 = self.b2 + self.lr * plateau * rpe * self.trace_b2
        self.W1 = self.W1 + self.lr * plateau * rpe * self.trace_W1 * mask1
        self.b1 = self.b1 + self.lr * plateau * rpe * self.trace_b1

        dW1_local = state.T @ (sd_hidden * plateau)
        self.W1 = self.W1 + 0.2 * self.lr * dW1_local * mask1
        self.b1 = self.b1 + 0.2 * self.lr * torch.mean(sd_hidden * plateau, dim=0, keepdim=True)

        pred_error = max(-5.0, min(5.0, td_target - (h @ self.Wr + self.br).item()))
        self.Wr = self.Wr + self.pred_lr * h.T * pred_error
        self.br = self.br + self.pred_lr * pred_error

        return rpe

    def reset_traces(self):
        self.trace_W1.zero_(); self.trace_b1.zero_()
        self.trace_W2.zero_(); self.trace_b2.zero_()


# ===================================================================
# Experiment runners
# ===================================================================

def run_classification(config_name, cfg, epochs=50, batch_size=32):
    """Classification on digits. Track accuracy every epoch + time-to-threshold."""
    X_tr, y_tr, X_te, y_te, nc = load_digits_data()
    Y_tr = one_hot(y_tr, nc)
    arch = [64, 128, 64, nc]

    if cfg is None:
        model = BackpropClassifier(arch, lr=0.2, seed=7)
    else:
        model = BioClassifier(arch, lr=0.3, plateau_gain=3.0,
                               trace_decay=0.7, seed=7, **cfg)

    n = X_tr.shape[0]
    losses = []
    test_accs = []
    first_80 = None
    first_90 = None
    first_95 = None

    for ep in range(1, epochs + 1):
        idx = torch.randperm(n, device=DEVICE)
        ep_loss = 0.0; nb = 0
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
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


def _state_to_tensor_oh(s, n):
    """One-hot encode scalar state to GPU tensor."""
    t = torch.zeros(1, n, device=DEVICE)
    t[0, s] = 1.0
    return t


def _state_to_tensor_vec(s, scale):
    """Continuous state to GPU tensor."""
    return torch.tensor(s.reshape(1, -1) / scale, dtype=torch.float32, device=DEVICE)


def run_chain_rl(config_name, cfg, length=20, n_ep=600):
    """Chain MDP. Track per-episode rewards + acquisition speed."""
    if cfg is None:
        agent = BackpropRLAgent(length, 32, 2, lr=0.01, pred_lr=0.02, seed=7)
    else:
        agent = BioRLAgent(length, 32, 2, lr=0.01, plateau_gain=3.0,
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
        total_r = 0.0

        for t in range(length * 5):
            sv = _state_to_tensor_oh(s, length)
            action, pred_r = agent.act(sv, rng, epsilon=eps)
            s2, r, done = env.step(action)
            total_r += r

            if done:
                td_target = r
            else:
                next_val = agent.predict_value(_state_to_tensor_oh(s2, length))
                td_target = r + 0.95 * next_val

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
    """CartPole with tuned hyperparams. Track steps + time-to-threshold."""
    cp_scale = np.array([2.4, 4.0, 0.21, 4.0])

    if cfg is None:
        agent = BackpropRLAgent(4, 64, 2, lr=0.005, pred_lr=0.05, seed=7)
    else:
        agent = BioRLAgent(4, 64, 2, lr=0.005, plateau_gain=3.0,
                            trace_decay=0.9, pred_lr=0.05, seed=7, **cfg)

    env = CartPole(seed=0)
    rng = np.random.RandomState(42)
    ep_steps = []
    first_50 = None
    first_100 = None
    first_150 = None

    for ep in range(1, n_ep + 1):
        s = env.reset()
        agent.reset_traces()
        eps = max(0.01, 0.2 - 0.19 * ep / n_ep)
        steps = 0

        for t in range(500):
            sv = _state_to_tensor_vec(s, cp_scale)
            action, pred_r = agent.act(sv, rng, epsilon=eps)
            s2, r, done = env.step(action)
            steps += 1

            if done:
                td_target = -1.0
            else:
                next_val = agent.predict_value(_state_to_tensor_vec(s2, cp_scale))
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
# Plotting
# ===================================================================

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
    }

    # === Figure 1: Learning curves ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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
    fig.savefig("simplicity_curves.png", dpi=150)
    print("  Saved simplicity_curves.png")

    # === Figure 2: Performance + Acquisition speed ===
    all_names = list(all_configs.keys())

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

    def color_for(name):
        cfg = all_configs[name]
        if cfg is None: return '#1f77b4'
        nc = sum(cfg.values())
        if nc == 5: return '#d62728'
        if nc <= 2: return '#2ca02c'
        return '#9467bd'

    cs = [color_for(n) for n in all_names]

    # Top row: final performance
    ax = axes2[0, 0]
    vals = [results[n]["classification"]["final_acc"] for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Test Accuracy"); ax.set_title("Classification — Final Accuracy")
    ax.set_xlim([0, 1.05]); ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, vals):
        ax.text(min(val + 0.005, 1.0), bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', fontsize=7)

    ax = axes2[0, 1]
    vals = [results[n]["chain"]["final_avg"] for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Avg Reward (last 50)"); ax.set_title("Chain — Final Reward")
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes2[0, 2]
    vals = [results[n]["cartpole"]["final_avg"] for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Avg Steps (last 50)"); ax.set_title("CartPole — Final Steps")
    ax.grid(True, alpha=0.3, axis='x')

    # Bottom row: acquisition speed (lower = faster = better)
    max_ep_class = 50; max_ep_chain = 600; max_ep_cart = 2000

    ax = axes2[1, 0]
    vals = [results[n]["classification"]["first_90"] or max_ep_class for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Epochs to 90% (fewer=better)")
    ax.set_title("Classification — Acquisition Speed")
    ax.set_xlim([0, max_ep_class + 5]); ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()
    for bar, val, name in zip(bars, vals, all_names):
        r90 = results[name]["classification"]["first_90"]
        label = str(val) if r90 else "never"
        ax.text(val - 0.5, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontsize=7, color='white', fontweight='bold')

    ax = axes2[1, 1]
    vals = [results[n]["chain"]["first_consistent"] or max_ep_chain for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Episodes to consistent (fewer=better)")
    ax.set_title("Chain — Acquisition Speed")
    ax.set_xlim([0, max_ep_chain + 20]); ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()
    for bar, val, name in zip(bars, vals, all_names):
        fc = results[name]["chain"]["first_consistent"]
        label = str(val) if fc else "never"
        ax.text(val - 2, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontsize=7, color='white', fontweight='bold')

    ax = axes2[1, 2]
    vals = [results[n]["cartpole"]["first_50"] or max_ep_cart for n in all_names]
    bars = ax.barh(range(len(all_names)), vals, color=cs, alpha=0.8)
    ax.set_yticks(range(len(all_names))); ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Episodes to avg>50 (fewer=better)")
    ax.set_title("CartPole — Acquisition Speed")
    ax.set_xlim([0, max_ep_cart + 50]); ax.grid(True, alpha=0.3, axis='x')
    ax.invert_xaxis()
    for bar, val, name in zip(bars, vals, all_names):
        f50 = results[name]["cartpole"]["first_50"]
        label = str(val) if f50 else "never"
        ax.text(val - 5, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right', fontsize=7, color='white', fontweight='bold')

    fig2.tight_layout()
    fig2.savefig("simplicity_full.png", dpi=150)
    print("  Saved simplicity_full.png")


# ===================================================================
# Main
# ===================================================================

def main():
    configs = {
        "Backprop":       None,
        "Bio (full)":     dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=True),
    }

    single_ablations = {
        "- SD":           dict(use_sd=False, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=True),
        "- BCP":          dict(use_sd=True, use_bcp=False, use_btsp=True, use_plateau=True, use_silent=True),
        "- BTSP":         dict(use_sd=True, use_bcp=True, use_btsp=False, use_plateau=True, use_silent=True),
        "- Plateau":      dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=False, use_silent=True),
        "- Silent":       dict(use_sd=True, use_bcp=True, use_btsp=True, use_plateau=True, use_silent=False),
    }

    minimal_combos = {
        "SD only":        dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=False, use_silent=False),
        "SD+Plateau":     dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=True, use_silent=False),
        "SD+BTSP":        dict(use_sd=True, use_bcp=False, use_btsp=True, use_plateau=False, use_silent=False),
        "SD+BTSP+Plat":   dict(use_sd=True, use_bcp=False, use_btsp=True, use_plateau=True, use_silent=False),
        "SD+Silent":      dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=False, use_silent=True),
        "SD+Plat+Sil":    dict(use_sd=True, use_bcp=False, use_btsp=False, use_plateau=True, use_silent=True),
    }

    all_configs = {**configs, **single_ablations, **minimal_combos}

    results = {}
    total_t0 = time.time()

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

    total_dt = time.time() - total_t0
    print(f"\nTotal time: {total_dt:.1f}s")

    # Summary table
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

    def fmt(v):
        return "—" if v is None else str(v)

    for name in all_configs:
        r = results[name]
        cfg = all_configs[name]
        nc = "BP" if cfg is None else str(sum(cfg.values()))

        cr = r["classification"]
        ch = r["chain"]
        cp = r["cartpole"]

        print(f"  {name:>18s}  {nc:>2s}  "
              f"{cr['final_acc']:5.3f} {fmt(cr['first_90']):>4s} {fmt(cr['first_95']):>4s}  "
              f"{ch['final_avg']:5.3f} {fmt(ch['first_solve']):>4s} {fmt(ch['first_consistent']):>4s}  "
              f"{cp['final_avg']:5.1f} {fmt(cp['first_50']):>5s} {fmt(cp['first_100']):>5s}")

    plot_results(results, all_configs)
    return results


if __name__ == "__main__":
    main()
