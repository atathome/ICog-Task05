import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from tinygrad import Tensor
from tinygrad.helpers import Context
import tinygrad.nn as nn_tg
from tinygrad import TinyJit

# ═══════════════════════════════════════════════════════════════
# Problem constants  (identical to learning-bayes2.py)
# ═══════════════════════════════════════════════════════════════
SIGMA   = 0.35
BEACONS = np.array([[4.0, 4.0], [4.0, -4.0], [-4.0, -4.0], [-4.0, 4.4]])
X_TRUE  = np.array([1.5, 0.7])
N_READ  = 80
BOX     = 7.0
K_B     = len(BEACONS)

# Pre-computed log-normalisation constants (avoid recomputing each call)
_LOG_NORM_SIGMA = float(np.log(SIGMA * np.sqrt(2 * np.pi)))
_LOG_K          = float(np.log(K_B))

# tinygrad constant for the beacon positions (not a parameter)
BEACONS_TG = Tensor(BEACONS.astype(np.float32))

# ═══════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════
def make_readings(seed=11):
    rng = np.random.default_rng(seed)
    who = rng.integers(0, K_B, N_READ)
    d   = np.linalg.norm(X_TRUE - BEACONS, axis=-1)[who]
    return d + rng.normal(0, SIGMA, N_READ)

# ═══════════════════════════════════════════════════════════════
# NumPy utilities (exact grid posterior, shared helpers)
# ═══════════════════════════════════════════════════════════════
def log_lik_np(x, readings):
    """x: (..., 2) → scalar per row (NumPy, for the exact grid)."""
    d  = np.linalg.norm(x[..., None, :] - BEACONS, axis=-1)   # (..., K)
    z  = (readings[None, :, None] - d[:, None, :]) / SIGMA     # (R, N, K)
    lp = -0.5 * z**2 - np.log(SIGMA * np.sqrt(2 * np.pi)) - np.log(K_B)
    return logsumexp(lp, axis=-1).sum(-1)                       # (R,)

def exact_grid(readings, step=0.04):
    g  = np.arange(-BOX, BOX + 1e-9, step)
    ll = np.empty((g.size, g.size))
    for i, gx in enumerate(g):
        ll[i] = log_lik_np(np.stack([np.full_like(g, gx), g], -1), readings)
    p = np.exp(ll - ll.max())
    return g, p / p.sum()

def image_centers():
    """The 8 symmetric images of X_TRUE under the near-square symmetry group."""
    a, b = X_TRUE
    return np.array([(a,b),(a,-b),(-a,b),(-a,-b),(b,a),(b,-a),(-b,a),(-b,-a)])

def assign(pts, centers):
    d = np.linalg.norm(pts[:, None, :] - centers[None], axis=-1)
    return d.argmin(1), d.min(1)

def masses(samples, centers):
    idx, _ = assign(samples, centers)
    return np.bincount(idx, minlength=len(centers)) / len(samples)

def tv(p, q):
    return 0.5 * np.abs(p - q).sum()

# ═══════════════════════════════════════════════════════════════
# Differentiable log-likelihood in tinygrad
# ═══════════════════════════════════════════════════════════════
def log_lik_tg(x, readings_tg):
    """
    Differentiable bag-of-readings log-likelihood.
    x           : Tensor (R, 2)   — candidate positions
    readings_tg : Tensor (N,)     — observed (pooled) distances
    Returns     : Tensor (R,)
    """
    R = x.shape[0]
    N = readings_tg.shape[0]

    # Euclidean distances to every beacon: (R, K)
    diff = x.reshape(R, 1, 2) - BEACONS_TG.reshape(1, K_B, 2)
    d    = (diff ** 2).sum(axis=-1).sqrt()

    # Expand for all readings × beacons: (R, N, K)
    r_e = readings_tg.reshape(1, N, 1).expand(R, N, K_B)
    d_e = d.reshape(R, 1, K_B).expand(R, N, K_B)

    z  = (r_e - d_e) / SIGMA
    lp = -0.5 * z ** 2 - _LOG_NORM_SIGMA - _LOG_K   # (R, N, K)
    return lp.logsumexp(axis=-1).sum(axis=-1)         # (R,)


# ═══════════════════════════════════════════════════════════════
# MLP building blocks  (plain tensors, no nn.Module needed)
# ═══════════════════════════════════════════════════════════════
def _kaiming(fan_in, fan_out):
    """He-uniform initialisation for ReLU networks."""
    limit = float(np.sqrt(6.0 / fan_in))
    t = Tensor(np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32))
    t.is_param_()
    return t

def _zeros(n):
    t = Tensor(np.zeros(n, dtype=np.float32))
    t.is_param_()
    return t

class MLP:
    """
    Feed-forward network: ReLU activations in hidden layers, linear output.
    dims: list of ints, e.g. [2, 64, 64, 1].
    """
    def __init__(self, dims):
        self.Ws = [_kaiming(dims[i], dims[i+1]) for i in range(len(dims) - 1)]
        self.bs = [_zeros(dims[i+1])             for i in range(len(dims) - 1)]

    def __call__(self, x):
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            x = x @ W + b
            if i < len(self.Ws) - 1:
                x = x.relu()
        return x

    def parameters(self):
        return [p for pair in zip(self.Ws, self.bs) for p in pair]


# ═══════════════════════════════════════════════════════════════
# 1. EXPLICIT VI — Mixture of Gaussians
# ═══════════════════════════════════════════════════════════════
class MoGVI:
    """
    Variational family:
        q(x; θ) = Σ_k π_k · N(x | μ_k, diag(σ_k²))

    Learnable parameters (all marked is_param_):
        log_w     : (K,)   unnormalised log-weights
        means     : (K, 2) component means
        log_sigma : (K, 2) log standard deviations (σ = exp(log_sigma))

    ELBO = E_q[log p*(x)] + H(q)
         = E_q[log L(r|x)] - E_q[log q(x)] + const
    Estimated by sampling S points from each of the K components
    (reparameterisation trick: x_{k,s} = μ_k + σ_k ⊙ ε,  ε ~ N(0,I)).
    """
    def __init__(self, K=8):
        centers = image_centers().astype(np.float32)
        noise   = np.random.randn(K, 2).astype(np.float32) * 0.3

        self.K = K
        self.log_w     = Tensor(np.zeros(K, dtype=np.float32))
        self.log_w.is_param_()
        self.means     = Tensor(centers + noise)
        self.means.is_param_()
        self.log_sigma = Tensor(np.zeros((K, 2), dtype=np.float32))  # σ=1 init
        self.log_sigma.is_param_()

    def parameters(self):
        return [self.log_w, self.means, self.log_sigma]

    # ── sample via reparameterisation ─────────────────────────────────────
    def sample(self, S):
        """
        Draw S samples from each of the K components.
        Returns Tensor (K*S, 2), differentiable w.r.t. means and log_sigma.
        """
        K = self.K
        eps   = Tensor.randn(K, S, 2)                                # fixed noise
        sigma = self.log_sigma.exp().reshape(K, 1, 2).expand(K, S, 2)
        mu    = self.means.reshape(K, 1, 2).expand(K, S, 2)
        return (mu + sigma * eps).reshape(K * S, 2)                   # (K*S, 2)

    # ── log q(x) for arbitrary x (fully differentiable) ──────────────────
    def log_q(self, x):
        """
        Compute log q(x) for a batch of points.
        x : Tensor (N, 2) → Tensor (N,)
        """
        N, K = x.shape[0], self.K
        # Normalised log-weights: log π_k
        log_pi = self.log_w - self.log_w.exp().sum().log()           # (K,)

        # Broadcast to (N, K, 2)
        xe = x.reshape(N, 1, 2).expand(N, K, 2)
        mu = self.means.reshape(1, K, 2).expand(N, K, 2)
        ls = self.log_sigma.reshape(1, K, 2).expand(N, K, 2)

        # log N(x | μ_k, σ_k²) summed over 2 dims → (N, K)
        log_gauss = (
            -0.5 * ((xe - mu) / ls.exp()) ** 2
            - ls
            - float(0.5 * np.log(2 * np.pi))
        )
        log_comp = log_gauss.sum(axis=-1) + log_pi.reshape(1, K).expand(N, K)
        return log_comp.logsumexp(axis=-1)                            # (N,)

    # ── ELBO (fully differentiable Monte-Carlo estimate) ─────────────────
    def elbo(self, readings_tg, S=64):
        x  = self.sample(S)                            # (K*S, 2)
        ll = log_lik_tg(x, readings_tg)               # (K*S,)  — log p*(x)
        lq = self.log_q(x)                             # (K*S,)  — log q(x)
        # Soft box constraint: heavy penalty for samples escaping [-BOX, BOX]^2
        pen = ((x.abs() - BOX).relu() ** 2).sum(axis=-1) * 500.0
        return (ll - lq - pen).mean()

    # ── sample for evaluation (NumPy, respects learned weights) ──────────
    def get_samples(self, n=5000):
        log_pi_np = self.log_w.numpy()
        log_pi_np -= np.log(np.exp(log_pi_np).sum())
        pi_np    = np.exp(log_pi_np)
        mu_np    = self.means.numpy()
        sigma_np = np.exp(self.log_sigma.numpy())
        counts   = np.round(pi_np * n).astype(int)
        counts[-1] = n - counts[:-1].sum()
        parts = []
        for k in range(self.K):
            eps = np.random.randn(max(counts[k], 0), 2)
            parts.append(eps * sigma_np[k] + mu_np[k])
        return np.clip(np.concatenate(parts, axis=0), -BOX, BOX)

def train_mog(readings_tg, n_steps=3000, lr=3e-3, S=64):
    """Train a MoG variational distribution by maximising the ELBO."""
    mog = MoGVI(K=8)
    opt = nn_tg.optim.Adam(mog.parameters(), lr=lr)
    hist = []

    @Context(TRAINING=1)
    @TinyJit
    def train_step():
        opt.zero_grad()
        elbo = mog.elbo(readings_tg, S=S)
        (-elbo).backward()
        opt.step()

        return elbo
    
    with Context(TRAINING=1):
        for step in range(n_steps):
            elbo = train_step()
            if step % 700 == 0:
                v = float(elbo.numpy())
                hist.append((step, v))
                print(f"  [MoG] step {step:4d}  ELBO = {v:+.3f}")
    return mog, hist


# ═══════════════════════════════════════════════════════════════
# 2. IMPLICIT VI — Adversarial Variational Bayes
#    with density-ratio estimation
# ═══════════════════════════════════════════════════════════════
class Generator:
    """
    Maps latent z ∈ ℝ^LATENT_DIM → x ∈ (-BOX, BOX)² via tanh squeeze.
    Architecture: [LATENT_DIM, H, H, 2], ReLU hidden, tanh*BOX output.
    """
    LATENT_DIM = 8

    def __init__(self, hidden=64):
        self.net = MLP([self.LATENT_DIM, hidden, hidden, 2])

    def __call__(self, z):
        return self.net(z).tanh() * BOX

    def sample(self, n):
        """Draw n posterior samples from q_φ."""
        return self(Tensor.randn(n, self.LATENT_DIM))

    def parameters(self):
        return self.net.parameters()


class Discriminator:
    """
    Estimates the log density-ratio T_ψ(x) ≈ log[q_φ(x) / p(x)].
    Architecture: [2, H, H, 1], ReLU hidden, linear output.

    Trained as a binary classifier:
        q-sample  (G_φ(z), z~N)  →  label 1
        prior sample (x~Uniform) →  label 0

    By Bayes' theorem, the optimal classifier satisfies
        logit D*(x) = log[q(x) / p(x)] = T*(x).
    """
    def __init__(self, hidden=64):
        self.net = MLP([2, hidden, hidden, 1])

    def __call__(self, x):
        return self.net(x).reshape(-1)   # (N,)

    def parameters(self):
        return self.net.parameters()

def train_avb(readings_tg, n_steps=50, lr=1e-3, batch=256, n_disc=5):
    """
    Adversarial Variational Bayes training loop.

    Alternating updates:
      Discriminator (n_disc steps):
        Minimise BCE between q-samples (label=1) and prior samples (label=0).
        → T_ψ learns to output log q(x)/p(x).

      Generator (1 step):
        Maximise ELBO ≈ E_z[log L(r|G(z))] - E_z[T_ψ(G(z))]
        Gradients flow back through G via the reparameterisation trick.
    """
    gen    = Generator(hidden=64)
    disc   = Discriminator(hidden=64)
    opt_g  = nn_tg.optim.Adam(gen.parameters(),  lr=lr)
    opt_d  = nn_tg.optim.Adam(disc.parameters(), lr=lr)

    gen_hist  = []   # (step, ELBO estimate)
    disc_hist = []   # (step, discriminator BCE loss)

    @TinyJit
    def disc_step():
        opt_d.zero_grad()
        for _ in range(n_disc):
            x_q = gen.sample(batch).detach()
            x_p = Tensor.rand(batch, 2) * (2*BOX) - BOX
            t_q = disc(x_q)
            t_p = disc(x_p)
            d_loss = -(t_q.logsigmoid().mean() + (-t_p).logsigmoid().mean())
            d_loss.backward()
            opt_d.step()
        return d_loss

    @TinyJit
    def gen_step():
        opt_g.zero_grad()
        x_q = gen.sample(batch)
        ll = log_lik_tg(x_q, readings_tg)
        t_q = disc(x_q)
        g_loss = -(ll - t_q).mean()
        g_loss.backward()
        opt_g.step()
        return g_loss

    with Context(TRAINING=1):
        for step in range(n_steps):

            # ── Discriminator update ──────────────────────────────────────
            # detach generator output: gradients stay in disc parameters only.
            d_loss_val = None
            for _ in range(n_disc):
                d_loss = disc_step()
                d_loss_val = float(d_loss.numpy())

            g_loss = gen_step()

            if step % 100 == 0:
                g_val = float(-g_loss.numpy())
                gen_hist.append((step, g_val))
                disc_hist.append((step, d_loss_val))
                print(
                    f"  [AVB] step {step:4d}"
                    f"  ELBO≈ {g_val:+.3f}"
                    f"  disc_BCE= {d_loss_val:.4f}"
                )

    return gen, disc, gen_hist, disc_hist


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(42)
    readings_np = make_readings(11)
    readings_tg = Tensor(readings_np.astype(np.float32))   # shared constant

    centers = image_centers()
    labels  = [f"({c[0]:+.1f},{c[1]:+.1f})" for c in centers]

    # ── exact posterior on a fine grid ───────────────────────────────────
    print("Computing exact grid posterior …")
    g, pgrid = exact_grid(readings_np)
    gx, gy   = np.meshgrid(g, g, indexing="ij")
    pts      = np.stack([gx.ravel(), gy.ravel()], -1)
    idx, _   = assign(pts, centers)
    exact    = np.bincount(idx, weights=pgrid.ravel(), minlength=8)
    print("Exact mode masses:", dict(zip(labels, np.round(exact, 3))))

    # ── Explicit VI: MoG ─────────────────────────────────────────────────
    print("\n=== Explicit VI (Mixture of Gaussians) ===")
    mog, mog_hist = train_mog(readings_tg, n_steps=22000, lr=3e-3, S=64)
    mog_samples   = mog.get_samples(5000)
    m_mog         = masses(mog_samples, centers)
    print(f"\nMoG-VI TV vs exact = {tv(m_mog, exact):.3f}")
    print("MoG-VI masses:", dict(zip(labels, np.round(m_mog, 3))))

    # Learned MoG parameters
    pi_np    = np.exp(mog.log_w.numpy())
    pi_np   /= pi_np.sum()
    mu_np    = mog.means.numpy()
    sigma_np = np.exp(mog.log_sigma.numpy())
    print("\nLearned MoG parameters:")
    for k in range(8):
        print(f"  k={k}  π={pi_np[k]:.3f}  μ={mu_np[k]}  σ={sigma_np[k]}")

    # ── Implicit VI: AVB ─────────────────────────────────────────────────
    print("\n=== Implicit VI (Adversarial Variational Bayes) ===")
    gen, disc, gen_hist, disc_hist = train_avb(
        readings_tg, n_steps=3000, lr=1e-3, batch=256, n_disc=5
    )
    avb_samples = gen.sample(5000).numpy()
    m_avb       = masses(avb_samples, centers)
    print(f"\nAVB-VI TV vs exact = {tv(m_avb, exact):.3f}")
    print("AVB-VI masses:", dict(zip(labels, np.round(m_avb, 3))))

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"{'Method':<24}  TV vs exact")
    print("─" * 60)
    print(f"{'MoG-VI  (explicit)':<24}  {tv(m_mog, exact):.3f}")
    print(f"{'AVB-VI  (implicit DRE)':<24}  {tv(m_avb, exact):.3f}")
    print(f"{'Exact (grid)':<24}  —")
    print("─" * 60)

    # ═══════════════════════════════════════════════════════════
    # Plotting  (2 × 3 layout)
    # ═══════════════════════════════════════════════════════════
    INK, INK2  = "#0b0b0b", "#52514e"
    SURF, GRID = "#fcfcfb", "#e5e4e0"
    BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#9b59b6"

    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "font.size": 10,
    })
    cmap = LinearSegmentedColormap.from_list("seq", [SURF, "#9ec3f0", BLUE, "#0e3a73"])

    img = pgrid.T          # shape (Ng, Ng); rows = y axis
    ext = [-BOX, BOX, -BOX, BOX]
    lim = 6.5

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))

    def draw_spatial(ax, title):
        """Draw the exact posterior contour + beacon markers on ax."""
        ax.contour(g, g, img, levels=[img.max() * 0.05],
                   colors=[INK2], linewidths=0.6, alpha=0.7)
        ax.plot(*BEACONS.T, "^", color=INK, ms=7, ls="none", label="beacon")
        ax.plot(*X_TRUE, "o", mfc="none", mec=INK, mew=1.5, ms=11,
                ls="none", label="true pos")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # (0,0) Exact posterior heatmap
    a = axes[0, 0]
    a.imshow(img, origin="lower", extent=ext, cmap=cmap,
             norm=PowerNorm(0.5), interpolation="bilinear")
    draw_spatial(a, "Exact posterior  (grid, √ colour scale)")
    a.legend(loc="lower left", fontsize=8, frameon=False)

    # (0,1) MoG-VI samples
    a = axes[0, 1]
    a.scatter(*mog_samples[::5].T, s=2, color=ORANGE, alpha=0.5)
    # Overlay MoG component ellipses (1-sigma)
    theta = np.linspace(0, 2*np.pi, 120)
    for k in range(8):
        cx, cy = mu_np[k]
        sx, sy = sigma_np[k]
        a.plot(cx + sx*np.cos(theta), cy + sy*np.sin(theta),
               color=ORANGE, lw=0.8, alpha=0.7)
    draw_spatial(a, f"Explicit VI — MoG  (TV = {tv(m_mog, exact):.3f})")

    # (0,2) AVB-VI samples
    a = axes[0, 2]
    a.scatter(*avb_samples[::5].T, s=2, color=AQUA, alpha=0.5)
    draw_spatial(a, f"Implicit VI — AVB  (TV = {tv(m_avb, exact):.3f})")

    # (1,0) MoG ELBO training curve
    a = axes[1, 0]
    if mog_hist:
        s_vals, e_vals = zip(*mog_hist)
        a.plot(s_vals, e_vals, color=ORANGE, lw=1.5)
        a.set_xlabel("optimisation step"); a.set_ylabel("ELBO")
        a.set_title("MoG-VI — ELBO during training", loc="left", fontsize=10)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    # (1,1) AVB training curves (dual y-axis)
    a = axes[1, 1]
    if gen_hist:
        g_steps, g_vals = zip(*gen_hist)
        d_steps, d_vals = zip(*disc_hist)
        a.plot(g_steps, g_vals, color=AQUA, lw=1.5, label="ELBO≈ (gen)")
        a.set_xlabel("optimisation step")
        a.set_ylabel("ELBO≈", color=AQUA)
        a.tick_params(axis="y", colors=AQUA)
        ax2 = a.twinx()
        ax2.plot(d_steps, d_vals, color=PURPLE, lw=1.5, alpha=0.7, label="BCE (disc)")
        ax2.set_ylabel("disc BCE", color=PURPLE)
        ax2.tick_params(axis="y", colors=PURPLE)
        a.set_title("AVB — generator & discriminator curves", loc="left", fontsize=10)
        for sp in ("top",):
            a.spines[sp].set_visible(False)
        # Combined legend
        lines = [
            plt.Line2D([0], [0], color=AQUA,   lw=1.5, label="ELBO≈ (gen)"),
            plt.Line2D([0], [0], color=PURPLE, lw=1.5, label="BCE (disc)"),
        ]
        a.legend(handles=lines, frameon=False, fontsize=8, loc="upper right")

    # (1,2) Mode mass comparison bar chart
    a = axes[1, 2]
    xs = np.arange(8)
    w  = 0.26
    a.bar(xs - w, exact, w * 0.9, color=BLUE,   label="exact (grid)")
    a.bar(xs,     m_mog, w * 0.9, color=ORANGE, label=f"MoG-VI (TV={tv(m_mog,exact):.2f})")
    a.bar(xs + w, m_avb, w * 0.9, color=AQUA,   label=f"AVB-VI (TV={tv(m_avb,exact):.2f})")
    a.set_xticks(xs)
    a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    a.set_ylabel("posterior mass in mode")
    a.set_title("Mode masses: exact vs VI", loc="left", fontsize=10)
    a.yaxis.grid(True, color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    a.tick_params(axis="y", length=0)
    a.legend(frameon=False, fontsize=9)

    fig.tight_layout(pad=1.5)
    out_path = "vi_trilateration.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nPlot saved → {out_path}")
