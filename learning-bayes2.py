import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

SIGMA = 0.35
BEACONS = np.array([[4.0, 4.0], [4.0, -4.0], [-4.0, -4.0], [-4.0, 4.4]])  # last one nudged off the square
X_TRUE = np.array([1.5, 0.7])
N_READ = 80
BOX = 7.0                       # uniform prior on [-BOX, BOX]^2
K = len(BEACONS)


def dists(x):
    """x: (..., 2) -> distances to every beacon (..., K)."""
    return np.linalg.norm(x[..., None, :] - BEACONS, axis=-1)


def make_readings(seed):
    rng = np.random.default_rng(seed)
    who = rng.integers(0, K, N_READ)                         # which beacon spoke: thrown away
    return dists(X_TRUE)[who] + rng.normal(0, SIGMA, N_READ)


def log_lik(x, readings):
    """x: (R, 2) -> (R,). Bag-of-readings likelihood, beacon identity summed out."""
    d = dists(x)                                             # (R, K)
    z = (readings[None, :, None] - d[:, None, :]) / SIGMA    # (R, n, K)
    lp = -0.5 * z ** 2 - np.log(SIGMA * np.sqrt(2 * np.pi)) - np.log(K)
    return logsumexp(lp, axis=-1).sum(-1)


def log_post(x, readings):
    inside = (np.abs(x) <= BOX).all(-1)
    return np.where(inside, log_lik(x, readings), -np.inf)


# ------------------------------------------------------------------ exact posterior on a grid
def exact_grid(readings, step=0.04):
    g = np.arange(-BOX, BOX + 1e-9, step)
    ll = np.empty((g.size, g.size))                          # ll[ix, iy]
    for i, gx in enumerate(g):
        ll[i] = log_lik(np.stack([np.full_like(g, gx), g], -1), readings)
    p = np.exp(ll - ll.max())
    return g, p / p.sum()


# ------------------------------------------------------------------ samplers
def rw_sampler(betas, n_iters, swap, seed, base_sd=0.18, sd_cap=2.0, readings=None):
    """Vectorised random-walk Metropolis over R replicas; replica k targets L(x)^betas[k] * prior.
    swap=False -> R independent plain-Metropolis chains (all betas should be 1).
    swap=True  -> parallel tempering: neighbouring replicas exchange states."""
    rng = np.random.default_rng(seed)
    R = len(betas)
    x = rng.uniform(-BOX, BOX, size=(R, 2))
    lp = log_post(x, readings)
    sd = np.minimum(base_sd / np.sqrt(betas), sd_cap)
    trace = np.empty((n_iters, R, 2))
    acc = np.zeros(R)
    sw_try = np.zeros(R - 1)
    sw_acc = np.zeros(R - 1)
    for it in range(n_iters):
        cand = x + rng.normal(size=x.shape) * sd[:, None]
        lp_c = log_post(cand, readings)
        with np.errstate(invalid="ignore"):
            ok = np.log(rng.uniform(size=R)) < betas * (lp_c - lp)
        x = np.where(ok[:, None], cand, x)
        lp = np.where(ok, lp_c, lp)
        acc += ok
        if swap:
            for k in range(it % 2, R - 1, 2):
                sw_try[k] += 1
                if np.log(rng.uniform()) < (betas[k] - betas[k + 1]) * (lp[k + 1] - lp[k]):
                    x[[k, k + 1]] = x[[k + 1, k]]
                    lp[[k, k + 1]] = lp[[k + 1, k]]
                    sw_acc[k] += 1
        trace[it] = x
    return trace, acc / n_iters, sw_acc / np.maximum(sw_try, 1)


# ------------------------------------------------------------------ mode bookkeeping
def image_centers():
    a, b = X_TRUE
    return np.array([(a, b), (a, -b), (-a, b), (-a, -b), (b, a), (b, -a), (-b, a), (-b, -a)])


def assign(points, centers):
    d = np.linalg.norm(points[:, None, :] - centers[None], axis=-1)
    return d.argmin(1), d.min(1)


def masses_from_samples(samples, centers):
    idx, _ = assign(samples, centers)
    return np.bincount(idx, minlength=len(centers)) / len(samples)


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


if __name__ == "__main__":
    readings = make_readings(11)
    centers = image_centers()
    labels = [f"({c[0]:+.1f},{c[1]:+.1f})" for c in centers]

    # ---- ground truth
    g, pgrid = exact_grid(readings)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], -1)
    idx, dmin = assign(pts, centers)
    exact = np.bincount(idx, weights=pgrid.ravel(), minlength=8)
    print("exact mode masses:", dict(zip(labels, np.round(exact, 3))))
    print(f"exact mass within 0.8 of an image centre: {pgrid.ravel()[dmin < 0.8].sum():.3f}")

    n_iters, burn = 30000, 3000
    R = 14

    # ---- plain Metropolis, R independent chains (same cost as PT)
    tr_plain, acc_plain, _ = rw_sampler(np.ones(R), n_iters, swap=False, seed=1, readings=readings)
    plain = tr_plain[burn:].reshape(-1, 2)
    m_plain = masses_from_samples(plain, centers)
    per_chain = [np.unique(assign(tr_plain[burn::50, r], centers)[0]).size for r in range(R)]
    print(f"plain MH acceptance {acc_plain.mean():.2f}; distinct modes visited per chain: {per_chain}")
    print("plain MH masses:  ", dict(zip(labels, np.round(m_plain, 3))), f" TV vs exact = {tv(m_plain, exact):.3f}")

    # ---- parallel tempering
    betas = np.geomspace(1.0, 0.004, R)
    tr_pt, acc_pt, sw = rw_sampler(betas, n_iters, swap=True, seed=2, readings=readings)
    cold = tr_pt[burn:, 0]
    m_pt = masses_from_samples(cold, centers)
    lab_cold = assign(cold, centers)[0]
    switches = int((lab_cold[1:] != lab_cold[:-1]).sum())
    print(f"PT move acceptance (cold..hot): {np.round(acc_pt, 2)}")
    print(f"PT swap acceptance (cold..hot): {np.round(sw, 2)}")
    print(f"PT cold chain switched mode {switches} times in {len(cold)} iterations")
    print("PT masses:        ", dict(zip(labels, np.round(m_pt, 3))), f" TV vs exact = {tv(m_pt, exact):.3f}")

    # ---- figure (single-hue sequential for magnitude; categorical slots 1-3 for the three methods)
    INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4e0"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 10})
    cmap = LinearSegmentedColormap.from_list("seq", [SURF, "#9ec3f0", BLUE, "#0e3a73"])
    ext = [-BOX, BOX, -BOX, BOX]
    img = pgrid.T                                            # rows = y
    lim = 6.0

    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5))

    def spatial(a, title):
        a.contour(g, g, img, levels=[img.max() * 0.05], colors=[INK2], linewidths=0.6, alpha=0.7)
        a.plot(*BEACONS.T, "^", color=INK, ms=8, ls="none", label="beacon")
        a.plot(*X_TRUE, "o", mfc="none", mec=INK, mew=1.5, ms=11, ls="none", label="true position")
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
        a.set_title(title, loc="left", fontsize=11)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    a = ax[0, 0]
    a.imshow(img, origin="lower", extent=ext, cmap=cmap, norm=PowerNorm(0.5), interpolation="bilinear")
    spatial(a, "Exact posterior over device position (grid, sqrt colour scale)")
    a.legend(loc="lower left", fontsize=8, frameon=False)

    a = ax[0, 1]
    a.plot(*plain[::25].T, ".", color=ORANGE, ms=2.5, alpha=0.5, ls="none")
    spatial(a, f"Plain Metropolis, {R} independent chains: each stays in one mode")

    a = ax[1, 0]
    a.plot(*cold[::10].T, ".", color=AQUA, ms=2.5, alpha=0.5, ls="none")
    spatial(a, f"Parallel tempering, cold chain: crosses between modes ({switches} switches)")

    a = ax[1, 1]
    w = 0.26
    xs = np.arange(8)
    a.bar(xs - w, exact, w * 0.9, color=BLUE, label="exact (grid)")
    a.bar(xs, m_plain, w * 0.9, color=ORANGE, label=f"plain Metropolis (TV {tv(m_plain, exact):.2f})")
    a.bar(xs + w, m_pt, w * 0.9, color=AQUA, label=f"parallel tempering (TV {tv(m_pt, exact):.2f})")
    a.set_xticks(xs); a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    a.set_ylabel("posterior mass of each mode")
    a.set_title("Mass in each of the 8 symmetric images of the true position", loc="left", fontsize=11)
    a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.tick_params(axis="y", length=0)
    a.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig("trilateration.png", dpi=120)
