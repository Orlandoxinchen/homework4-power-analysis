"""
Homework 4, Part 1 — numerical verification and figures for the analytical problems.
Orlando Leng
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=200)
    fig.savefig(f"{FIGDIR}/{name}.pdf")
    plt.close(fig)
    print(f"  saved {FIGDIR}/{name}.png")

# ================= Problem 1: normal distribution =================
mu, sd = 0.8, 2.5
print("=" * 62); print("PROBLEM 1 — Normal(mu = 0.8%, sigma = 2.5%)"); print("=" * 62)
z_a = (0 - mu) / sd;  p_a = stats.norm.cdf(z_a)
z_b1, z_b2 = (-1 - mu) / sd, (3 - mu) / sd
p_b = stats.norm.cdf(z_b2) - stats.norm.cdf(z_b1)
z_c = (4 - mu) / sd;  p_c = 1 - stats.norm.cdf(z_c)
z_d = stats.norm.ppf(0.10); x_d = mu + z_d * sd
print(f"  a) z = {z_a:.4f}            P(X<0)        = {p_a:.4f}  ({100*p_a:.2f}%)")
print(f"  b) z = {z_b1:.4f}, {z_b2:.4f}   P(-1<X<3)     = {p_b:.4f}  ({100*p_b:.2f}%)")
print(f"  c) z = {z_c:.4f}             P(X>4)        = {p_c:.4f}  ({100*p_c:.2f}%)")
print(f"  d) z = {z_d:.4f}            10th pctile   = {x_d:.4f}%")

xs = np.linspace(mu - 4*sd, mu + 4*sd, 600)
pdf = stats.norm.pdf(xs, mu, sd)
panels = [("(a)  $P(X<0)$ = %.4f" % p_a,            xs < 0,                 "////"),
          ("(b)  $P(-1<X<3)$ = %.4f" % p_b,         (xs > -1) & (xs < 3),   "...."),
          ("(c)  $P(X>4)$ = %.4f" % p_c,            xs > 4,                 "\\\\\\\\"),
          ("(d)  bottom 10%%: $x$ = %.3f%%" % x_d,  xs < x_d,               "xxxx")]
fig, axs = plt.subplots(2, 2, figsize=(6.6, 3.5))
for ax, (title, mask, hatch) in zip(axs.ravel(), panels):
    ax.plot(xs, pdf, color="0.1", lw=1.0)
    ax.fill_between(xs, 0, pdf, where=mask, facecolor="0.80",
                    edgecolor="0.25", hatch=hatch, lw=0.0)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_yticks([]); ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, pdf.max()*1.18)
    ax.axvline(mu, color="0.45", lw=0.6, ls=":")
    ax.tick_params(labelsize=7.5, length=2.5)
    ax.set_xlabel("daily return (%)", fontsize=7.5, labelpad=1)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
fig.tight_layout(h_pad=1.0, w_pad=1.4)
save(fig, "p1_normal")

# ================= Problem 2: MLE =================
data = np.array([1.2, 0.8, 1.5, 2.1, 0.9, 1.7, 1.3, 1.1, 1.6, 1.4])
n, S = len(data), data.sum()
lam_hat = n / S
print(); print("=" * 62); print("PROBLEM 2 — Exponential MLE"); print("=" * 62)
print(f"  n = {n},  sum(x) = {S:.1f},  xbar = {S/n:.3f}")
print(f"  lambda_hat = n / sum(x) = {n}/{S:.1f} = {lam_hat:.6f} per second")
print(f"  scipy check (fit, loc fixed at 0): {1/stats.expon.fit(data, floc=0)[1]:.6f}")
print(f"  log-likelihood at lambda_hat: {n*np.log(lam_hat) - lam_hat*S:.6f}")

lams = np.linspace(0.25, 1.6, 500)
ll = n * np.log(lams) - lams * S
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.plot(lams, ll, color="0.1", lw=1.2)
ax.axvline(lam_hat, color="0.4", lw=0.8, ls=(0, (4, 2)))
ax.plot([lam_hat], [n*np.log(lam_hat) - lam_hat*S], "o", ms=4, color="0.1")
ax.annotate(r"$\hat{\lambda}=%.4f$" % lam_hat,
            xy=(lam_hat, n*np.log(lam_hat) - lam_hat*S), xytext=(8, -10),
            textcoords="offset points", fontsize=8.5)
ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"log-likelihood  $\ell(\lambda)$")
ax.tick_params(length=2.5)
save(fig, "p2_loglik")

# ================= Problem 3: sample size =================
z_a_, z_b_ = 1.645, 0.84
def nreq(p0, p1):
    return (z_a_ + z_b_)**2 * (p0*(1-p0) + p1*(1-p1)) / (p1 - p0)**2
p0 = 0.08
n_c, n_d = nreq(p0, 0.10), nreq(p0, 0.09)
print(); print("=" * 62); print("PROBLEM 3 — Sample size per group"); print("=" * 62)
print(f"  (z_a+z_b)^2 = ({z_a_}+{z_b_})^2 = {(z_a_+z_b_)**2:.6f}")
print(f"  c) 8% -> 10%:  n = {n_c:.4f}  ->  {int(np.ceil(n_c)):,} per group")
print(f"  d) 8% ->  9%:  n = {n_d:.4f}  ->  {int(np.ceil(n_d)):,} per group")
print(f"  ratio d/c = {n_d/n_c:.3f}")

p1s = np.linspace(0.085, 0.14, 300)
ns = [nreq(p0, p) for p in p1s]
fig, ax = plt.subplots(figsize=(4.8, 2.6))
ax.plot(100*(p1s - p0), ns, color="0.1", lw=1.2)
for lbl, p1, nn in [("(c)  8%$\\to$10%", 0.10, n_c), ("(d)  8%$\\to$9%", 0.09, n_d)]:
    ax.plot([100*(p1-p0)], [nn], "o", ms=4, color="0.1")
    ax.annotate(f"{lbl}\nn = {int(np.ceil(nn)):,}", xy=(100*(p1-p0), nn),
                xytext=(10, 6), textcoords="offset points", fontsize=8)
ax.set_xlabel("effect size to detect, $p_1-p_0$ (percentage points)")
ax.set_ylabel("required $n$ per group")
ax.set_yscale("log"); ax.tick_params(length=2.5)
ax.grid(True, which="both", color="0.9", lw=0.5)
save(fig, "p3_samplesize")

# ================= Part B summary figure =================
rng = np.random.default_rng(42)
sizes = np.arange(10, 201, 5)
z_alpha = stats.norm.ppf(0.95)
shift = (53 - 50) / (10 / np.sqrt(sizes))
pw = 1 - stats.norm.cdf(z_alpha - shift)
fig, ax = plt.subplots(figsize=(4.8, 2.6))
ax.plot(sizes, pw, color="0.1", lw=1.2, label="theoretical power")
sim = {30: 0.4964, 60: 0.7529, 120: 0.9493}
ax.plot(list(sim), list(sim.values()), "s", ms=4.5, mfc="white",
        mec="0.1", mew=1.0, label="simulated (10,000 reps)")
for k, v in sim.items():
    ax.annotate(f"{v:.3f}", xy=(k, v), xytext=(5, -9), textcoords="offset points", fontsize=8)
ax.axhline(0.8, color="0.45", lw=0.7, ls=":")
ax.annotate("power = 0.80", xy=(188, 0.805), ha="right", fontsize=8, color="0.3")
ax.set_xlabel("sample size $n$"); ax.set_ylabel("power")
ax.set_ylim(0, 1.03); ax.legend(loc="lower right", fontsize=8); ax.tick_params(length=2.5)
save(fig, "p2b_power_vs_n")
print("\nAll Part 1 figures written.")
