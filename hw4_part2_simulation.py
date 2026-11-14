"""
Homework 4, Part 2 — Hypothesis Testing Simulation and Power Analysis
Orlando Leng

Explores the relationship between Type I error, Type II error, and the true
parameter value by Monte Carlo simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed
np.random.seed(42)

FIGDIR = "figures"

# ============================================================
# PART A: Simulating the Type I Error Rate
# ============================================================
#
# Test:
#   H0: mu = 50
#   H1: mu != 50
#
# When H0 is TRUE, we should reject it approximately alpha
# proportion of the time (5% here).

mu_0 = 50
sigma = 10
n = 30
alpha = 0.05
n_simulations = 10000

# 1. Generate n_simulations samples from N(mu_0, sigma^2), each of size n.
#    One row per simulated experiment.
samples_A = np.random.normal(mu_0, sigma, size=(n_simulations, n))

# 2a. Sample mean of each simulated experiment.
xbar_A = samples_A.mean(axis=1)

# 2b. z-statistic. sigma is known, so the standard error is sigma / sqrt(n).
z_A = (xbar_A - mu_0) / (sigma / np.sqrt(n))

# 2c. Two-sided p-value: P(|Z| > |z|) = 2 * (1 - Phi(|z|)).
p_A = 2 * (1 - stats.norm.cdf(np.abs(z_A)))

# 3. Reject H0 when p < alpha.
reject_A = p_A < alpha

# 4. Empirical Type I error rate.
type1_rate = reject_A.mean()

# 5. Compare with the theoretical alpha.
#    Monte Carlo standard error of a proportion: sqrt(a(1-a)/N).
mc_se = np.sqrt(alpha * (1 - alpha) / n_simulations)

print("=" * 62)
print("PART A — Empirical Type I error rate")
print("=" * 62)
print(f"  simulations            : {n_simulations:,}")
print(f"  rejections             : {reject_A.sum():,}")
print(f"  empirical Type I error : {type1_rate:.4f}")
print(f"  theoretical alpha      : {alpha:.4f}")
print(f"  difference             : {type1_rate - alpha:+.4f}")
print(f"  Monte Carlo std. error : {mc_se:.4f}")
print(f"  95% MC interval        : [{alpha - 1.96*mc_se:.4f}, {alpha + 1.96*mc_se:.4f}]")
print(f"  inside that interval?  : {abs(type1_rate - alpha) < 1.96 * mc_se}")

# ============================================================
# PART B: Simulating Type II Error and Statistical Power
# ============================================================
#
# Test:
#   H0: mu = 50
#   H1: mu > 50          (right-tailed)
#
# The true population mean is actually 53, so H0 is false and
# failing to reject it is a Type II error.

mu_0 = 50
true_mu = 53
sigma = 10
alpha = 0.05
n_simulations = 10000

sample_sizes = [30, 60, 120]
results_B = []

for n in sample_sizes:
    # 1. Samples drawn from the TRUE distribution N(53, 10^2).
    samples = np.random.normal(true_mu, sigma, size=(n_simulations, n))

    # 2. Sample mean and z-statistic, still measured against mu_0 = 50.
    xbar = samples.mean(axis=1)
    z = (xbar - mu_0) / (sigma / np.sqrt(n))

    # 3. Right-tailed test: p = P(Z > z).
    p = 1 - stats.norm.cdf(z)

    # 4. Reject H0 when p < alpha.
    reject = p < alpha

    # 5-6. power = proportion rejected; beta = proportion failing to reject.
    power = reject.mean()
    beta = 1 - power

    # Theoretical power for a known-sigma right-tailed z-test:
    #   power = 1 - Phi(z_alpha - (mu_true - mu_0) / (sigma/sqrt(n)))
    z_alpha = stats.norm.ppf(1 - alpha)
    shift = (true_mu - mu_0) / (sigma / np.sqrt(n))
    power_theory = 1 - stats.norm.cdf(z_alpha - shift)

    # 7. Store.
    results_B.append((n, beta, power, power_theory))

print()
print("=" * 62)
print("PART B — Type II error and power (true mu = 53, H1: mu > 50)")
print("=" * 62)
print(f"{'Sample Size':>12}{'Beta':>12}{'Power':>12}{'Power (theory)':>18}")
print("-" * 62)
for n, beta, power, power_theory in results_B:
    print(f"{n:>12}{beta:>12.4f}{power:>12.4f}{power_theory:>18.4f}")
print("-" * 62)
print("  check: power + beta = 1 for every row, by construction.")

# ============================================================
# PART C: Power Curve Analysis
# ============================================================
#
# Null hypothesis fixed, TRUE mean varied.
#
# Test:
#   H0: mu = 50
#   H1: mu != 50         (two-sided, alpha = 0.05)

mu_0 = 50
sigma = 10
n = 30
alpha = 0.05

# 51 values so that mu = 50 is included exactly
mu_true_values = np.linspace(45, 55, 51)
n_simulations = 2000

power_values = []

for true_mu in mu_true_values:
    # 1. Samples from N(true_mu, sigma^2).
    samples = np.random.normal(true_mu, sigma, size=(n_simulations, n))

    # 2-3. Sample mean and z-statistic against mu_0.
    xbar = samples.mean(axis=1)
    z = (xbar - mu_0) / (sigma / np.sqrt(n))

    # 4. Two-sided p-value.
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    # 5-6. Proportion of rejections.
    power_values.append((p < alpha).mean())

power_values = np.array(power_values)

# Theoretical two-sided power curve, for reference.
z_crit = stats.norm.ppf(1 - alpha / 2)
delta = (mu_true_values - mu_0) / (sigma / np.sqrt(n))
power_theory_curve = (1 - stats.norm.cdf(z_crit - delta)) + stats.norm.cdf(-z_crit - delta)

idx50 = int(np.where(np.isclose(mu_true_values, 50))[0][0])
print()
print("=" * 62)
print("PART C — Power curve (n = 30, two-sided)")
print("=" * 62)
print(f"  rejection rate at true mu = 50 : {power_values[idx50]:.4f}  (should be ~alpha = 0.05)")
for m in (47, 48, 49, 51, 52, 53):
    j = int(np.where(np.isclose(mu_true_values, m))[0][0])
    print(f"  power at true mu = {m:<2d}           : {power_values[j]:.4f}"
          f"   (theory {power_theory_curve[j]:.4f})")

# ============================================================
# Plot the Power Curve
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.8,
})

plt.figure(figsize=(8, 5))

plt.plot(mu_true_values, power_values, color="black", lw=1.3,
         marker="o", ms=3.2, mfc="white", mew=0.7,
         label="Simulated rejection rate (2,000 reps)")
plt.plot(mu_true_values, power_theory_curve, color="0.55", lw=1.1, ls=(0, (5, 2)),
         label="Theoretical power")

plt.axvline(mu_0, color="0.25", lw=0.9, ls=":", label=r"$\mu_0 = 50$")
plt.axhline(alpha, color="0.25", lw=0.9, ls="-.", label=r"$\alpha = 0.05$")

plt.xlabel("True Population Mean")
plt.ylabel("Probability of Rejecting H0")
plt.title("Power Curve")
plt.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="0.7", fontsize=9)
plt.grid(True, color="0.85", lw=0.6)
plt.ylim(-0.03, 1.05)
plt.xlim(44.8, 55.2)
plt.tight_layout()

import os
os.makedirs(FIGDIR, exist_ok=True)
plt.savefig(os.path.join(FIGDIR, "power_curve.png"), dpi=200)
plt.savefig(os.path.join(FIGDIR, "power_curve.pdf"))
print(f"\nSaved {FIGDIR}/power_curve.png and {FIGDIR}/power_curve.pdf")
plt.show()
