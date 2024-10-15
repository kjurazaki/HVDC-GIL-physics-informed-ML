import math
from scipy.special import ellipk, ellipe
import numpy as np
import matplotlib.pyplot as plt


def compute_v_fn(y_FN):
    k_squared = (2 * math.sqrt(1 - y_FN**2)) / (1 + math.sqrt(1 - y_FN**2))
    K_k = ellipk(k_squared)
    E_k = ellipe(k_squared)
    v_y = (2**-0.5) * (
        math.sqrt(1 + math.sqrt(1 - y_FN**2))
        * (E_k - (1 - math.sqrt(1 - y_FN**2)) * K_k)
    )
    return v_y


def compute_emittance(
    lambda_FN=0.005,
    a_FN=7.7e-11,
    beta_FN=68,
    E=1,
    W_w=1.77,
):
    """
    Based on the Fowler-Nordheim equation, compute the emittance of cathode.
    Reference: winter, transient
    """
    # Fowler-Nordheim constants
    alpha_FN = 1.54e-6
    b_FN = 6.83e9
    # Elementary charge in Coulombs
    elementary_charge = 1.602176634e-19
    # Equation (8)
    y_FN = (3.79e-5 * math.sqrt(E)) / W_w
    # Equation (7)
    v_FN = compute_v_fn(y_FN)
    exponent = -(v_FN * b_FN * W_w ** (3 / 2)) / (beta_FN * E)
    emittance = (
        alpha_FN
        * lambda_FN
        * a_FN
        * (beta_FN * E) ** 2
        / W_w
        / elementary_charge
        * math.exp(exponent)
        * W_w
    )

    return emittance


def compute_emittance_with_intervals(
    lambda_FN_range=(0.005, 10),
    beta_FN_range=(68, 800),
    W_w_range=(1.77, 3.95),
    a_FN_range=(7.7e-11, 1.5e-7),
    E=1,
    num_samples=1000,
):
    """
    Compute the emittance using uniformly spaced values within the specified intervals.
    """
    lambda_FN_values = np.linspace(lambda_FN_range[0], lambda_FN_range[1], num_samples)
    beta_FN_values = np.linspace(beta_FN_range[0], beta_FN_range[1], num_samples)
    W_w_values = np.linspace(W_w_range[0], W_w_range[1], num_samples)
    a_FN_values = np.linspace(a_FN_range[0], a_FN_range[1], num_samples)

    emittance_values = [
        compute_emittance(
            lambda_FN=lambda_FN,
            a_FN=a_FN,
            beta_FN=beta_FN,
            E=E,
            W_w=W_w,
        )
        for lambda_FN, a_FN, beta_FN, W_w in zip(
            lambda_FN_values, a_FN_values, beta_FN_values, W_w_values
        )
    ]

    return emittance_values


# comuting
elementary_charge = 1.602176634e-19

emittance = compute_emittance(
    E=4e6, a_FN=7.5e-8, lambda_FN=5.0025, W_w=2.86, beta_FN=200
)
print(f"Computed emittance: {emittance}")
print(f"Computed current: {emittance * elementary_charge}")
emittance_interval = compute_emittance_with_intervals(
    E=6e6,
    lambda_FN_range=(0.005, 0.005),
    beta_FN_range=(68, 68),
    W_w_range=(1.77, 3.95),
    a_FN_range=(7.7e-11, 7.7e-11),
    num_samples=100,
)
print(f"Computed min emittance: {min(emittance_interval)}")

# Create a distribution plot of emittance_interval values
plt.figure(figsize=(10, 6))
plt.hist(emittance_interval, bins=30, edgecolor="black")
plt.title("Distribution of Emittance Values")
plt.xlabel("Emittance")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)

# Add some statistical information
plt.axvline(
    np.mean(emittance_interval),
    color="r",
    linestyle="dashed",
    linewidth=2,
    label=f"Mean: {np.mean(emittance_interval):.2e}",
)
plt.axvline(
    np.median(emittance_interval),
    color="g",
    linestyle="dashed",
    linewidth=2,
    label=f"Median: {np.median(emittance_interval):.2e}",
)

plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean emittance: {np.mean(emittance_interval):.2e}")
print(f"Median emittance: {np.median(emittance_interval):.2e}")
print(f"Standard deviation: {np.std(emittance_interval):.2e}")
