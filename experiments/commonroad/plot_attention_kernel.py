"""
Visualise the human attention kernel used for velocity-error evolution.

The attention kernel f(d) determines how quickly the human's velocity
error kappa decays toward 0 as a function of following distance:

    kappa_{t+1} = kappa_t * (1 - b_kappa * f(d_t))

A higher f(d) means more attention → faster correction of perception bias.

Currently implemented:
    RBF:      f(d) = exp(-d^2 / (2 * sigma^2))
    Bump:     f(d) = exp(-1 / (1 - (d/r)^2))  for |d| < r, else 0
    Linear:   f(d) = max(0, 1 - d / d_max)
    Sigmoid:  f(d) = 1 / (1 + exp(k * (d - d_mid)))

Usage:
    python experiments/commonroad/plot_attention_kernel.py
    python experiments/commonroad/plot_attention_kernel.py --sigma 25 --b_kappa 0.01
    python experiments/commonroad/plot_attention_kernel.py -o kernel.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  Kernel functions
# ---------------------------------------------------------------------------

def rbf_kernel(d, sigma=25.0):
    """Gaussian RBF: peaks at d=0, decays with distance."""
    return np.exp(-d ** 2 / (2.0 * sigma ** 2))


def bump_kernel(d, radius=30.0):
    """Smooth bump: compactly supported, C-infinity."""
    f = np.zeros_like(d)
    mask = np.abs(d) < radius
    r = (d[mask] / radius) ** 2
    f[mask] = np.exp(-1.0 / (1.0 - r))
    return f


def linear_kernel(d, d_max=50.0):
    """Linear decay: 1 at d=0, 0 at d=d_max."""
    return np.clip(1.0 - d / d_max, 0.0, 1.0)


def sigmoid_kernel(d, d_mid=20.0, k=0.2):
    """Sigmoid: high attention when close, drops off around d_mid."""
    return 1.0 / (1.0 + np.exp(k * (d - d_mid)))


KERNELS = {
    "rbf": rbf_kernel,
    "bump": bump_kernel,
    "linear": linear_kernel,
    "sigmoid": sigmoid_kernel,
}


# ---------------------------------------------------------------------------
#  Kappa evolution simulation
# ---------------------------------------------------------------------------

def simulate_kappa_evolution(kernel_fn, kappa_0=0.3, b_kappa=0.01,
                             distance=15.0, n_steps=200, dt=0.1):
    """Simulate kappa evolution at a fixed distance."""
    kappas = [kappa_0]
    for _ in range(n_steps):
        f = kernel_fn(np.array([distance]))[0]
        kappas.append(kappas[-1] * (1.0 - b_kappa * f))
    return np.array(kappas)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    d = np.linspace(0, 80, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel 1: Kernel shape f(d) ---
    ax = axes[0, 0]
    styles = {"rbf": "-", "bump": "--", "linear": "-.", "sigmoid": ":"}
    colours = {"rbf": "steelblue", "bump": "#e74c3c",
               "linear": "#2ecc71", "sigmoid": "#f39c12"}

    for name, fn in KERNELS.items():
        if name == "rbf":
            f = fn(d, sigma=args.sigma)
        elif name == "bump":
            f = fn(d, radius=args.bump_radius)
        elif name == "linear":
            f = fn(d, d_max=args.linear_dmax)
        elif name == "sigmoid":
            f = fn(d, d_mid=args.sigmoid_dmid, k=args.sigmoid_k)
        ax.plot(d, f, styles[name], color=colours[name],
                linewidth=2, label=name)

    ax.set_xlabel("following distance $d$ [m]")
    ax.set_ylabel("attention $f(d)$")
    ax.set_title("Attention kernel shape")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.axvline(args.d_safe, color="red", linestyle="--", alpha=0.4,
               label=f"$d_{{safe}}$={args.d_safe}")
    ax.axvline(args.d_target, color="green", linestyle="--", alpha=0.4,
               label=f"$d^*$={args.d_target}")
    ax.legend(fontsize=8)

    # --- Panel 2: Drift rate b_kappa * f(d) ---
    ax = axes[0, 1]
    for name, fn in KERNELS.items():
        if name == "rbf":
            f = fn(d, sigma=args.sigma)
        elif name == "bump":
            f = fn(d, radius=args.bump_radius)
        elif name == "linear":
            f = fn(d, d_max=args.linear_dmax)
        elif name == "sigmoid":
            f = fn(d, d_mid=args.sigmoid_dmid, k=args.sigmoid_k)
        ax.plot(d, args.b_kappa * f, styles[name], color=colours[name],
                linewidth=2, label=name)

    ax.set_xlabel("following distance $d$ [m]")
    ax.set_ylabel("drift rate $b_\\kappa \\cdot f(d)$")
    ax.set_title(f"Per-step kappa decay rate ($b_\\kappa$={args.b_kappa})")
    ax.legend(fontsize=9)

    # --- Panel 3: Kappa evolution at fixed distance ---
    ax = axes[1, 0]
    time = np.arange(0, args.n_steps + 1) * 0.1

    for dist in [10, 15, 20, 30, 50]:
        kappas = simulate_kappa_evolution(
            lambda d_arr: rbf_kernel(d_arr, sigma=args.sigma),
            kappa_0=args.kappa_0, b_kappa=args.b_kappa,
            distance=dist, n_steps=args.n_steps)
        ax.plot(time, kappas, linewidth=1.5,
                label=f"d={dist}m", alpha=0.8)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("$\\kappa(t)$")
    ax.set_title(f"RBF kappa evolution at fixed distances "
                 f"($\\kappa_0$={args.kappa_0}, $\\sigma$={args.sigma})")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)

    # --- Panel 4: Kappa evolution comparison across kernels ---
    ax = axes[1, 1]
    dist_fixed = 15.0

    for name, fn in KERNELS.items():
        if name == "rbf":
            kern = lambda d_arr: rbf_kernel(d_arr, sigma=args.sigma)
        elif name == "bump":
            kern = lambda d_arr: bump_kernel(d_arr, radius=args.bump_radius)
        elif name == "linear":
            kern = lambda d_arr: linear_kernel(d_arr, d_max=args.linear_dmax)
        elif name == "sigmoid":
            kern = lambda d_arr: sigmoid_kernel(d_arr, d_mid=args.sigmoid_dmid,
                                                k=args.sigmoid_k)
        kappas = simulate_kappa_evolution(
            kern, kappa_0=args.kappa_0, b_kappa=args.b_kappa,
            distance=dist_fixed, n_steps=args.n_steps)
        ax.plot(time, kappas, styles[name], color=colours[name],
                linewidth=2, label=name)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("$\\kappa(t)$")
    ax.set_title(f"Kappa evolution at d={dist_fixed}m "
                 f"($\\kappa_0$={args.kappa_0})")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9)

    fig.suptitle("Human Attention Kernel Analysis", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.out}")
    else:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise attention kernels for kappa evolution")
    p.add_argument("--sigma", type=float, default=25.0,
                   help="RBF sigma (default: 25.0)")
    p.add_argument("--b_kappa", type=float, default=0.01,
                   help="Drift rate (default: 0.01)")
    p.add_argument("--kappa_0", type=float, default=0.3,
                   help="Initial kappa (default: 0.3)")
    p.add_argument("--n_steps", type=int, default=200,
                   help="Number of steps to simulate (default: 200)")
    p.add_argument("--d_safe", type=float, default=15.0)
    p.add_argument("--d_target", type=float, default=20.0)
    p.add_argument("--bump_radius", type=float, default=30.0,
                   help="Bump kernel radius (default: 30.0)")
    p.add_argument("--linear_dmax", type=float, default=50.0,
                   help="Linear kernel max distance (default: 50.0)")
    p.add_argument("--sigmoid_dmid", type=float, default=20.0,
                   help="Sigmoid midpoint distance (default: 20.0)")
    p.add_argument("--sigmoid_k", type=float, default=0.2,
                   help="Sigmoid steepness (default: 0.2)")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Save figure to file")
    return p.parse_args()


if __name__ == "__main__":
    main()
