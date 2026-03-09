"""
Biologically Plausible Learning — Correct Implementation from insights.md

Models:
  1. Backprop    — standard global gradient (baseline)
  2. BioNetwork  — full biological model: SD Residuals + BCP + BTSP

    The three-factor rule:  ΔW_ij = η · Eligibility_j(t) · Mismatch_i(t)

    For supervised: Mismatch = top-down target - somatic activity
    For RL:         Mismatch = RPE projected through feedback to local SD residuals
                    (each neuron gets a vectorized error specific to its contribution)

    Key mechanisms:
      - SD Residual: local somato-dendritic mismatch at every neuron
      - BCP: separate E/I populations, balance disruption = error signal
      - BTSP: eligibility traces, plateau potentials for temporal credit
      - Silent synapse gating for stability

Experiments:
  1. XOR            2. N-bit Parity     3. Multi-armed Bandit
  4. Chain MDP      5. CartPole
"""

import numpy as np
import argparse
import time
from typing import Dict


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def tanh_act(x): return np.tanh(x)
def tanh_deriv(x): return 1.0 - np.tanh(x)**2
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_deriv(x):
    s = sigmoid(x); return s * (1.0 - s)
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. Backprop Network (baseline)
# ---------------------------------------------------------------------------

class BackpropNetwork:
    """Standard MLP. Sigmoid output for supervised, linear for RL value."""
    def __init__(self, sizes, lr=0.5, linear_out=False, seed=42):
        rng = np.random.RandomState(seed)
        self.lr = lr; self.lin = linear_out; self.nl = len(sizes) - 1
        self.W = [rng.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0/sizes[i])
                  for i in range(self.nl)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(self.nl)]

    def forward(self, x):
        a = [x]; p = [x]
        for i in range(self.nl):
            z = a[-1] @ self.W[i] + self.b[i]; p.append(z)
            if i < self.nl - 1:
                a.append(tanh_act(z))
            else:
                a.append(z if self.lin else sigmoid(z))
        return a, p

    def train_step(self, x, y):
        a, p = self.forward(x); o = a[-1]
        loss = np.mean((y - o)**2)
        od = np.ones_like(o) if self.lin else sigmoid_deriv(p[-1])
        d = (o - y) * od
        for i in reversed(range(self.nl)):
            self.W[i] -= self.lr * (a[i].T @ d / x.shape[0])
            self.b[i] -= self.lr * np.mean(d, axis=0, keepdims=True)
            if i > 0: d = (d @ self.W[i].T) * tanh_deriv(p[i])
        return loss

    def predict(self, x): return self.forward(x)[0][-1]


# ---------------------------------------------------------------------------
# 2. BioNetwork — Full biological model from insights.md
# ---------------------------------------------------------------------------

class BioNetwork:
    """
    Full implementation of the three-factor learning rule from insights.md.

    Architecture per hidden layer:
      - Excitatory population (E): feedforward drive, somatic activity
      - Inhibitory population (I): tracks E activity, maintains balance
      - Feedback connections (B): project error to dendritic compartment
      - Eligibility traces: BTSP-style synaptic tags
      - Silent synapse mask: preferential plasticity on weak synapses

    Supervised mode:
      SD_Residual = top-down target (via feedback) - somatic activity
      ΔW = η · eligibility · SD_residual

    RL mode (policy gradient with biological three-factor rule):
      - Forward pass records Hebbian eligibility traces
      - Output = stochastic policy (action probabilities)
      - Reward prediction error (RPE) = actual - predicted reward
      - RPE projected through feedback → local SD residuals at each neuron
      - ΔW = η · eligibility · SD_residual (which encodes RPE locally)
      - Silent synapse gating prevents catastrophic forgetting
    """

    def __init__(self, sizes, lr=0.5, plateau_gain=3.0, trace_decay=0.8,
                 silent_thresh=0.5, ei_tau=0.05, seed=42):
        rng = np.random.RandomState(seed)
        self.lr = lr
        self.plateau_gain = plateau_gain
        self.trace_decay = trace_decay
        self.silent_thresh = silent_thresh
        self.ei_tau = ei_tau
        self.nl = len(sizes) - 1
        self.sizes = sizes

        # Forward (excitatory) weights
        self.W = [rng.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0/sizes[i])
                  for i in range(self.nl)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(self.nl)]

        # Feedback connections: project from output space to each hidden layer
        # (models apical dendritic input — how top-down signals reach each neuron)
        out = sizes[-1]
        self.B = [rng.randn(out, sizes[i+1]) * np.sqrt(1.0/out)
                  for i in range(self.nl - 1)]

        # Inhibitory population: tracks excitatory activity (BCP)
        # Separate per-neuron inhibitory prediction
        self.I_pred = [np.zeros((1, sizes[i+1])) for i in range(self.nl)]

        # BTSP eligibility traces (per-synapse)
        self.traces = [np.zeros_like(w) for w in self.W]
        self.traces_b = [np.zeros_like(b) for b in self.b]

    def _silent_mask(self, W):
        """Silent synapse gating: weak synapses get full plasticity,
        strong synapses are partially protected."""
        return 1.0 / (1.0 + np.abs(W) / self.silent_thresh)

    def forward(self, x):
        a = [x]; p = [x]
        for i in range(self.nl):
            z = a[-1] @ self.W[i] + self.b[i]; p.append(z)
            if i < self.nl - 1:
                a.append(tanh_act(z))
            else:
                a.append(sigmoid(z))
        return a, p

    def _compute_sd_residuals(self, activations, pre_acts, error_at_output):
        """
        Compute local SD Residual at each layer.

        The output error (target - output, or RPE projected to output) is
        projected through feedback connections B to each hidden layer's
        dendritic compartment. The mismatch between this dendritic target
        and the somatic activity = the SD Residual.

        This gives each neuron a VECTORIZED error (signed, scaled) specific
        to its causal contribution — not just a global scalar.
        """
        residuals = [None] * self.nl

        # Output layer: direct error
        residuals[-1] = error_at_output * sigmoid_deriv(pre_acts[-1])

        # Hidden layers: project output error through feedback to dendrites
        for i in range(self.nl - 2, -1, -1):
            # Apical dendritic input: output error projected through feedback B
            dendritic_target = error_at_output @ self.B[i]

            # BCP: compute balance error between E and I
            ei_balance = activations[i+1] - self.I_pred[i]
            # Update inhibitory prediction (slow tracking)
            self.I_pred[i] += self.ei_tau * np.mean(
                activations[i+1] - self.I_pred[i], axis=0, keepdims=True)

            # SD Residual = dendritic signal + balance disruption
            # The dendritic signal gives direction (from task error)
            # The balance disruption gives magnitude (from local state)
            # dis-inhibition (E > I) → LTP, excess inhibition (I > E) → LTD
            sd_residual = (dendritic_target + 0.1 * ei_balance)
            sd_residual *= tanh_deriv(pre_acts[i+1])

            residuals[i] = sd_residual

        return residuals

    def train_step(self, x, y):
        """Supervised learning: direct target → SD residual → three-factor update."""
        a, p = self.forward(x)
        o = a[-1]; n = x.shape[0]
        loss = np.mean((y - o)**2)

        # Output error
        error = y - o

        # Compute local SD residuals at every layer
        residuals = self._compute_sd_residuals(a, p, error)

        # Three-factor update at each layer
        for i in range(self.nl):
            # Plateau gain: models dendritic calcium spike amplification
            local_err = residuals[i] * self.plateau_gain

            # Hebbian term: pre × post-error
            dW = a[i].T @ local_err / n
            db = np.mean(local_err, axis=0, keepdims=True)

            # Eligibility trace accumulation (BTSP: tag active synapses)
            self.traces[i] = self.trace_decay * self.traces[i] + (1 - self.trace_decay) * dW
            self.traces_b[i] = self.trace_decay * self.traces_b[i] + (1 - self.trace_decay) * db

            # Silent synapse mask (stability)
            mask = self._silent_mask(self.W[i])

            # Apply three-factor update: lr × eligibility × mismatch × mask
            # In supervised mode, we use immediate gradient + trace reinforcement
            self.W[i] += self.lr * (dW + 0.3 * self.traces[i]) * mask
            self.b[i] += self.lr * (db + 0.3 * self.traces_b[i])

        return loss

    def predict(self, x): return self.forward(x)[0][-1]

    def reset_traces(self):
        for i in range(self.nl):
            self.traces[i] *= 0
            self.traces_b[i] *= 0


# ---------------------------------------------------------------------------
# 3. BioRL — Biological RL agent (policy + reward prediction + three-factor)
# ---------------------------------------------------------------------------

class BioRLAgent:
    """
    Biological RL agent using the three-factor rule with SD Residuals.

    Key differences from backprop REINFORCE:
      1. Eligibility traces (BTSP) — credit propagates backward through time
         via decaying synaptic tags, NOT via value bootstrapping.
         This is TD(λ) with high λ vs backprop's TD(0).
      2. Adaptive plateau — surprise (large |RPE|) triggers high-gain updates
         modeling dendritic calcium spikes. Routine outcomes → small updates.
      3. SD Residuals — vectorized per-neuron error via feedback projection.
      4. BCP — E/I balance disruption as additional error signal.
    """

    def __init__(self, state_dim, hidden, n_actions, lr=0.01, plateau_gain=3.0,
                 trace_decay=0.85, silent_thresh=3.0, ei_tau=0.05,
                 pred_lr=0.01, seed=42):
        rng = np.random.RandomState(seed)
        self.lr = lr
        self.plateau_gain = plateau_gain
        self.trace_decay = trace_decay
        self.silent_thresh = silent_thresh
        self.ei_tau = ei_tau
        self.pred_lr = pred_lr
        self.n_actions = n_actions

        # Policy network
        self.W1 = rng.randn(state_dim, hidden) * np.sqrt(2.0/state_dim)
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.randn(hidden, n_actions) * np.sqrt(2.0/hidden)
        self.b2 = np.zeros((1, n_actions))

        # Reward predictor (dendritic expectation)
        self.Wr = rng.randn(hidden, 1) * np.sqrt(1.0/hidden)
        self.br = np.zeros((1, 1))

        # Feedback connections (apical pathway — fixed random, not W2.T)
        self.B = rng.randn(n_actions, hidden) * np.sqrt(1.0/n_actions)

        # BCP: inhibitory prediction
        self.I_pred = np.zeros((1, hidden))

        # BTSP eligibility traces
        self.trace_W1 = np.zeros_like(self.W1)
        self.trace_b1 = np.zeros_like(self.b1)
        self.trace_W2 = np.zeros_like(self.W2)
        self.trace_b2 = np.zeros_like(self.b2)

        # Adaptive plateau: running average of |RPE| for surprise detection
        self.rpe_ema = 0.1

        # Store last forward pass
        self.last_state = None
        self.last_hidden = None
        self.last_pre_h = None
        self.last_probs = None
        self.last_action = None

    def _silent_mask(self, W):
        return 1.0 / (1.0 + np.abs(W) / self.silent_thresh)

    def predict_value(self, state):
        z1 = state @ self.W1 + self.b1
        h = tanh_act(z1)
        return (h @ self.Wr + self.br).item()

    def act(self, state, rng, epsilon=0.0):
        """Forward pass + stochastic action + trace accumulation."""
        z1 = state @ self.W1 + self.b1
        h = tanh_act(z1)
        z2 = h @ self.W2 + self.b2
        probs = softmax(z2)

        if rng.random() < epsilon:
            action = rng.randint(self.n_actions)
        else:
            action = rng.choice(self.n_actions, p=probs.ravel())

        pred_reward = (h @ self.Wr + self.br).item()

        # Hebbian eligibility: pre × post-error (BTSP tagging)
        action_onehot = np.zeros_like(probs)
        action_onehot[0, action] = 1.0
        policy_error = action_onehot - probs

        # Layer 2 traces: straightforward Hebbian
        hebbian_W2 = h.T @ policy_error
        hebbian_b2 = policy_error.copy()

        # Layer 1 traces: use FEEDBACK B (biological), not W2.T (backprop)
        hidden_signal = policy_error @ self.B * tanh_deriv(z1)
        hebbian_W1 = state.T @ hidden_signal
        hebbian_b1 = hidden_signal.copy()

        # Accumulate eligibility traces (BTSP: decaying synaptic tags)
        self.trace_W2 = self.trace_decay * self.trace_W2 + hebbian_W2
        self.trace_b2 = self.trace_decay * self.trace_b2 + hebbian_b2
        self.trace_W1 = self.trace_decay * self.trace_W1 + hebbian_W1
        self.trace_b1 = self.trace_decay * self.trace_b1 + hebbian_b1

        # Store for learning
        self.last_state = state
        self.last_hidden = h
        self.last_pre_h = z1
        self.last_probs = probs
        self.last_action = action

        return action, pred_reward

    def learn(self, td_target, pred_reward):
        """
        Online three-factor update.

        RPE = td_target - predicted (dopamine-like).
        Adaptive plateau: surprising RPE → high-gain calcium spike.
        Traces carry credit backward in time (BTSP).
        SD residuals give per-neuron vectorized error.
        """
        h = self.last_hidden
        z1 = self.last_pre_h
        state = self.last_state

        # RPE (dopamine-like signal)
        rpe = td_target - pred_reward
        rpe = np.clip(rpe, -5.0, 5.0)

        # === Adaptive plateau: surprise-based gain ===
        # Large |RPE| relative to recent history → calcium spike → high gain
        # Small |RPE| (expected outcome) → routine, low gain
        surprise = abs(rpe) / (self.rpe_ema + 1e-6)
        plateau = 1.0 + (self.plateau_gain - 1.0) * min(surprise, 3.0) / 3.0
        # Update running RPE average
        self.rpe_ema = 0.95 * self.rpe_ema + 0.05 * abs(rpe)

        # === SD Residuals: project RPE to each neuron ===

        # Hidden layer: RPE projected through feedback B to dendrites
        action_onehot = np.zeros((1, self.n_actions))
        action_onehot[0, self.last_action] = 1.0
        rpe_signal = rpe * (action_onehot - self.last_probs)
        dendritic_input = rpe_signal @ self.B

        # BCP: E/I balance error
        ei_error = h - self.I_pred
        self.I_pred += self.ei_tau * (h - self.I_pred)

        # SD Residual at hidden layer: dendritic RPE + balance disruption
        sd_hidden = (dendritic_input + 0.1 * ei_error) * tanh_deriv(z1)

        # === Three-factor update ===
        mask2 = self._silent_mask(self.W2)
        mask1 = self._silent_mask(self.W1)

        # Output: lr × plateau × RPE × eligibility × mask
        self.W2 += self.lr * plateau * rpe * self.trace_W2 * mask2
        self.b2 += self.lr * plateau * rpe * self.trace_b2

        # Hidden: lr × plateau × RPE × eligibility × mask
        self.W1 += self.lr * plateau * rpe * self.trace_W1 * mask1
        self.b1 += self.lr * plateau * rpe * self.trace_b1

        # SD residual correction (per-neuron vectorized direction)
        dW1_local = state.T @ (sd_hidden * plateau) / state.shape[0]
        self.W1 += 0.2 * self.lr * dW1_local * mask1
        self.b1 += 0.2 * self.lr * np.mean(sd_hidden * plateau, axis=0, keepdims=True)

        # === Update reward predictor ===
        pred_error = np.clip(td_target - (h @ self.Wr + self.br), -5.0, 5.0)
        self.Wr += self.pred_lr * h.T * pred_error
        self.br += self.pred_lr * pred_error

        return rpe

    def reset_traces(self):
        self.trace_W1 *= 0; self.trace_b1 *= 0
        self.trace_W2 *= 0; self.trace_b2 *= 0


# ---------------------------------------------------------------------------
# Backprop RL agent (REINFORCE baseline for fair comparison)
# ---------------------------------------------------------------------------

class BackpropRLAgent:
    """Standard REINFORCE with baseline for fair RL comparison."""

    def __init__(self, state_dim, hidden, n_actions, lr=0.01,
                 pred_lr=0.01, seed=42):
        rng = np.random.RandomState(seed)
        self.lr = lr; self.pred_lr = pred_lr; self.n_actions = n_actions

        self.W1 = rng.randn(state_dim, hidden) * np.sqrt(2.0/state_dim)
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.randn(hidden, n_actions) * np.sqrt(2.0/hidden)
        self.b2 = np.zeros((1, n_actions))
        self.Wr = rng.randn(hidden, 1) * np.sqrt(1.0/hidden)
        self.br = np.zeros((1, 1))

        self.log_grads = []  # store (dW1, db1, dW2, db2) per step

    def predict_value(self, state):
        z1 = state @ self.W1 + self.b1; h = tanh_act(z1)
        return (h @ self.Wr + self.br).item()

    def act(self, state, rng, epsilon=0.0):
        z1 = state @ self.W1 + self.b1; h = tanh_act(z1)
        z2 = h @ self.W2 + self.b2; probs = softmax(z2)
        pred_r = (h @ self.Wr + self.br).item()

        if rng.random() < epsilon:
            action = rng.randint(self.n_actions)
        else:
            action = rng.choice(self.n_actions, p=probs.ravel())

        # Compute and store policy gradient for this step
        oh = np.zeros_like(probs); oh[0, action] = 1.0
        pe = oh - probs
        self._last_grad = (
            state.T @ (pe @ self.W2.T * tanh_deriv(z1)),  # dW1
            pe @ self.W2.T * tanh_deriv(z1),               # db1
            h.T @ pe, pe.copy(), h)                        # dW2, db2, h
        return action, pred_r

    def learn(self, td_target, pred_reward):
        """Online actor-critic update with TD error."""
        rpe = td_target - pred_reward
        rpe = np.clip(rpe, -2.0, 2.0)
        if hasattr(self, '_last_grad'):
            dW1, db1, dW2, db2, h = self._last_grad
            self.W2 += self.lr * rpe * dW2
            self.b2 += self.lr * rpe * db2
            self.W1 += self.lr * rpe * dW1
            self.b1 += self.lr * rpe * db1
            # Update value baseline
            pred_error = np.clip(td_target - (h @ self.Wr + self.br), -2.0, 2.0)
            self.Wr += self.pred_lr * h.T * pred_error
            self.br += self.pred_lr * pred_error
        return rpe

    def reset_traces(self):
        pass


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

class MultiArmedBandit:
    def __init__(self, k=10, drift=0.01, seed=42):
        self.rng = np.random.RandomState(seed)
        self.k = k; self.drift = drift; self.means = self.rng.randn(k)
    def reset(self): return 0
    def step(self, a):
        r = self.means[a] + self.rng.randn()
        self.means += self.drift * self.rng.randn(self.k)
        return 0, r, False

class ChainMDP:
    def __init__(self, length=10, slip=0.1, seed=42):
        self.L = length; self.slip = slip
        self.rng = np.random.RandomState(seed); self.s = 0
    def reset(self): self.s = 0; return self.s
    def step(self, a):
        if self.rng.random() < self.slip: a = 1 - a
        self.s = max(0, min(self.L-1, self.s + (1 if a else -1)))
        d = self.s == self.L-1; return self.s, (1.0 if d else 0.0), d

class CartPole:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed); self.state = None; self.reset()
    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4); return self.state.copy()
    def step(self, a):
        x, xd, th, thd = self.state
        f = 10.0 if a else -10.0; c, s = np.cos(th), np.sin(th)
        tmp = (f + 0.05*thd**2*s)/1.1
        tha = (9.8*s - c*tmp)/(0.5*(4/3 - 0.1*c**2/1.1))
        xa = tmp - 0.05*tha*c/1.1
        self.state = np.array([x+0.02*xd, xd+0.02*xa, th+0.02*thd, thd+0.02*tha])
        d = abs(self.state[0]) > 2.4 or abs(self.state[2]) > 12*np.pi/180
        return self.state.copy(), (0.0 if d else 1.0), d


# ---------------------------------------------------------------------------
# State encoders
# ---------------------------------------------------------------------------

def one_hot(s, n):
    oh = np.zeros((1, n)); oh[0, s] = 1.0; return oh

def vec_state(s):
    return s.reshape(1, -1) if len(s.shape) == 1 else s


# ---------------------------------------------------------------------------
# Supervised experiments
# ---------------------------------------------------------------------------

def run_xor(epochs=500, pev=50):
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    Y = np.array([[0],[1],[1],[0]], dtype=float)

    nets = {
        "Backprop": BackpropNetwork([2, 16, 1], lr=2.0, seed=7),
        "Bio":      BioNetwork([2, 16, 1], lr=1.5, plateau_gain=2.5, seed=7),
    }
    return _supervised_loop(nets, X, Y, epochs, 0.005, pev)


def run_parity(bits=3, epochs=3000, pev=300):
    N = 2**bits
    X = np.array([[(i>>b)&1 for b in range(bits)] for i in range(N)], dtype=float)
    Y = (X.sum(1, keepdims=True) % 2).astype(float)
    h = max(16, bits*8); nl = max(2, bits-1)
    sizes = [bits] + [h]*nl + [1]
    print(f"  bits={bits}, samples={N}, arch={sizes}")

    nets = {
        "Backprop": BackpropNetwork(sizes, lr=1.0, seed=7),
        "Bio":      BioNetwork(sizes, lr=0.5, plateau_gain=3.0, seed=7),
    }
    return _supervised_loop(nets, X, Y, epochs, 0.02, pev)


def _supervised_loop(nets, X, Y, epochs, thresh, pev):
    res = {}
    for name, net in nets.items():
        L = []; conv = None
        for ep in range(1, epochs+1):
            l = net.train_step(X, Y); L.append(l)
            if conv is None and l < thresh: conv = ep
            if ep % pev == 0 or ep == 1:
                print(f"  [{name:>10}] ep {ep:4d}  loss={l:.6f}")
        pr = net.predict(X)
        res[name] = {"losses": L, "converged": conv, "preds": pr}
        cs = f"epoch {conv}" if conv else "did not converge"
        print(f"  [{name:>10}] {cs} | preds: {pr.ravel().round(3)}\n")
    return res


# ---------------------------------------------------------------------------
# RL experiments
# ---------------------------------------------------------------------------

def _rl_loop(agents, env_fn, state_fn, n_episodes, max_steps,
             eps_start, eps_end, print_every, gamma=0.99, metric="reward",
             use_mc=False):
    """
    RL loop supporting both online TD and Monte Carlo returns.
    TD mode: learn at each step with r + γV(s') target.
    MC mode: collect episode, compute discounted returns, update per-step.
    """
    results = {}
    for name, agent in agents.items():
        env = env_fn()
        rng = np.random.RandomState(42)
        ep_vals = []; first_solve = None

        for ep in range(1, n_episodes + 1):
            s = env.reset()
            agent.reset_traces()
            eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / n_episodes)
            total_r = 0; steps = 0

            if use_mc:
                # MC mode: collect episode first, then learn
                ep_data = []  # (state_vec, action, reward, pred_r)
                for t in range(max_steps):
                    sv = state_fn(s)
                    action, pred_r = agent.act(sv, rng, epsilon=eps)
                    s2, r, done = env.step(action)
                    total_r += r; steps += 1
                    ep_data.append((sv, action, r, pred_r))
                    s = s2
                    if done: break

                # Compute discounted returns
                G = 0
                returns = []
                for _, _, r, _ in reversed(ep_data):
                    G = r + gamma * G
                    returns.insert(0, G)

                # Normalize returns for variance reduction
                returns = np.array(returns)
                if len(returns) > 1:
                    ret_std = returns.std() + 1e-8
                    returns = (returns - returns.mean()) / ret_std

                # Re-forward and learn for each step
                agent.reset_traces()
                for t, (sv, action, r, pred_r) in enumerate(ep_data):
                    # Re-build traces by re-forwarding
                    agent.act(sv, rng, epsilon=0.0)
                    agent.last_action = action  # override with actual action taken
                    agent.learn(returns[t], pred_r)
            else:
                # Online TD mode
                for t in range(max_steps):
                    sv = state_fn(s)
                    action, pred_r = agent.act(sv, rng, epsilon=eps)
                    s2, r, done = env.step(action)
                    total_r += r; steps += 1

                    if done:
                        td_target = r
                    else:
                        next_val = agent.predict_value(state_fn(s2))
                        td_target = r + gamma * next_val

                    agent.learn(td_target, pred_r)
                    s = s2
                    if done: break

            val = total_r if metric == "reward" else steps
            ep_vals.append(val)
            if first_solve is None and total_r > 0 and metric == "reward":
                first_solve = ep
            if ep % print_every == 0:
                recent = np.mean(ep_vals[-print_every:])
                print(f"  [{name:>10}] ep {ep:4d}  avg={recent:.2f}")

        results[name] = {
            "ep_values": ep_vals,
            "final_avg": np.mean(ep_vals[-50:]),
            "first_solve": first_solve,
        }
        fs = f"ep {first_solve}" if first_solve else "never"
        print(f"  [{name:>10}] first_solve={fs} | final_50={results[name]['final_avg']:.2f}\n")
    return results


def run_bandit(n_steps=3000, k=10, pev=500):
    """Bandit: single-step RL. Tests basic reward-modulated learning."""
    agents = {
        "Backprop": BackpropRLAgent(1, 32, k, lr=0.02, pred_lr=0.01, seed=7),
        "Bio":      BioRLAgent(1, 32, k, lr=0.02, plateau_gain=2.0,
                                trace_decay=0.5, seed=7),
    }
    state_fn = lambda s: np.ones((1, 1))

    results = {}
    for name, agent in agents.items():
        env = MultiArmedBandit(k=k, seed=0)
        rng = np.random.RandomState(42)
        rews = []; cum = 0.0
        for step in range(1, n_steps+1):
            sv = state_fn(0)
            action, pred_r = agent.act(sv, rng, epsilon=0.1)
            _, r, _ = env.step(action)
            cum += r
            # Bandit: no next state, TD target = reward directly
            agent.learn(r, pred_r)
            rews.append(cum/step)
            if step % pev == 0:
                print(f"  [{name:>10}] step {step:4d}  avg={cum/step:.3f}")
        results[name] = {"avg_rewards": rews, "final_avg": cum/n_steps}
        print(f"  [{name:>10}] final: {cum/n_steps:.3f}\n")
    return results


def run_chain(length=20, n_ep=800, pev=100):
    agents = {
        "Backprop": BackpropRLAgent(length, 32, 2, lr=0.01, pred_lr=0.02, seed=7),
        "Bio":      BioRLAgent(length, 32, 2, lr=0.01, plateau_gain=3.0,
                                trace_decay=0.95, pred_lr=0.02, seed=7),
    }
    return _rl_loop(agents,
        env_fn=lambda: ChainMDP(length=length, slip=0.1, seed=0),
        state_fn=lambda s: one_hot(s, length),
        n_episodes=n_ep, max_steps=length*5,
        eps_start=0.3, eps_end=0.05, print_every=pev, gamma=0.95)


def run_cartpole(n_ep=1000, pev=100):
    # Normalize CartPole state to similar scales
    cp_scale = np.array([2.4, 4.0, 0.21, 4.0])
    agents = {
        "Backprop": BackpropRLAgent(4, 32, 2, lr=0.001, pred_lr=0.1, seed=7),
        "Bio":      BioRLAgent(4, 32, 2, lr=0.001, plateau_gain=3.0,
                                trace_decay=0.8, pred_lr=0.1, seed=7),
    }
    return _rl_loop(agents,
        env_fn=lambda: CartPole(seed=0),
        state_fn=lambda s: vec_state(s) / cp_scale,
        n_episodes=n_ep, max_steps=200,
        eps_start=0.3, eps_end=0.02, print_every=pev, gamma=0.99,
        metric="steps")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_res: Dict[str, dict]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (no matplotlib)"); return

    colors = {"Backprop": "#1f77b4", "Bio": "#d62728"}
    n = len(all_res)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]

    for ax, (exp, res) in zip(axes, all_res.items()):
        for m, d in res.items():
            c = colors.get(m, None)
            if "losses" in d:
                ax.plot(d["losses"], label=m, alpha=0.8, color=c)
                ax.set_yscale("log"); ax.set_ylabel("MSE"); ax.set_xlabel("Epoch")
            elif "avg_rewards" in d:
                ax.plot(d["avg_rewards"], label=m, alpha=0.8, color=c)
                ax.set_ylabel("Avg Reward"); ax.set_xlabel("Step")
            elif "ep_values" in d:
                v = np.array(d["ep_values"]); w = min(20, len(v))
                ax.plot(np.convolve(v, np.ones(w)/w, 'valid'), label=m, alpha=0.8, color=c)
                ax.set_ylabel("Value (smoothed)"); ax.set_xlabel("Episode")
        ax.set_title(exp); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/Users/sbae703/Research/ML/local_max/results.png", dpi=150)
    print("  Saved results.png"); plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default="all",
                    choices=["all", "xor", "parity", "bandit", "chain", "cartpole"])
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    R = {}
    if args.experiment in ("all", "xor"):
        print("="*60); print("XOR"); print("="*60)
        R["XOR"] = run_xor()
    if args.experiment in ("all", "parity"):
        print("="*60); print(f"{args.bits}-BIT PARITY"); print("="*60)
        R["Parity"] = run_parity(bits=args.bits)
    if args.experiment in ("all", "bandit"):
        print("="*60); print("BANDIT"); print("="*60)
        R["Bandit"] = run_bandit()
    if args.experiment in ("all", "chain"):
        print("="*60); print(f"CHAIN (len={args.length})"); print("="*60)
        R["Chain"] = run_chain(length=args.length)
    if args.experiment in ("all", "cartpole"):
        print("="*60); print("CARTPOLE"); print("="*60)
        R["CartPole"] = run_cartpole()

    print("="*60); print("SUMMARY"); print("="*60)
    for exp, res in R.items():
        print(f"\n  {exp}:")
        for m, d in res.items():
            p = []
            if "converged" in d:
                c = d["converged"]
                p.append(f"converged={c}" if c else "NOT converged")
            if "first_solve" in d and d["first_solve"]:
                p.append(f"first_solve=ep{d['first_solve']}")
            if "final_avg" in d:
                p.append(f"final={d['final_avg']:.2f}")
            print(f"    {m:>10}: {' | '.join(p)}")

    if not args.no_plot and R:
        print(); plot_results(R)

if __name__ == "__main__":
    main()
