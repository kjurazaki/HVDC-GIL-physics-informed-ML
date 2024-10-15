import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

"""
Based on https://pysindy.readthedocs.io/en/latest/examples/3_original_paper/example.html
"""


def generate_data(
    N=1000, mus=[2.5, 2.75, 3, 3.25, 3.5, 3.75, 3.8, 3.85, 3.9, 3.95], eps=0.025
):
    x = [np.zeros((N, 2)) for i in range(len(mus))]
    for i, mu in enumerate(mus):
        x[i][0] = [0.5, mu]
        for k in range(1, N):
            x[i][k, 0] = np.maximum(
                np.minimum(
                    mu * x[i][k - 1, 0] * (1 - x[i][k - 1, 0])
                    + eps * np.random.randn(),
                    1.0,
                ),
                0.0,
            )
            x[i][k, 1] = mu
    return x


# Fit the model
def fit_model(x_train):
    optimizer = ps.STLSQ(threshold=0.1)
    library = ps.PolynomialLibrary(degree=5)
    model = ps.SINDy(optimizer=optimizer, feature_library=library, discrete_time=True)
    model.fit(x_train, multiple_trajectories=True)
    model.print()
    return model


def generate_test_data(N=1000, mus=np.arange(1, 4, 0.01), eps=0.025):
    x_test = np.zeros((mus.size * N, 2))
    idx = 0
    for mu in mus:
        xold = 0.5
        for i in range(N):
            xnew = np.maximum(
                np.minimum(mu * xold - mu * xold**2 + eps * np.random.randn(), 1.0),
                0.0,
            )
            xold = xnew
        xss = xnew
        for i in range(N):
            xnew = np.maximum(
                np.minimum(mu * xold - mu * xold**2 + eps * np.random.randn(), 1.0),
                0.0,
            )
            xold = xnew
            x_test[idx, 0] = xnew
            x_test[idx, 1] = mu
            idx += 1
            if np.abs(xnew - xss) < 0.001:
                break
    x_test = x_test[:idx]
    return x_test


def simulate_model(model, mus, N):
    x_sim = np.zeros((mus.size * N, 2))
    idx = 0
    for mu in mus:
        xss = model.simulate([0.5, mu], N)[-1]
        stop_condition = lambda x: np.abs(x[0] - xss[0]) < 0.001
        x = model.simulate(xss, N, stop_condition=stop_condition)
        idx_new = idx + x.shape[0]
        x_sim[idx:idx_new] = x
        idx = idx_new
    x_sim = x_sim[:idx]
    return x_sim


# Plot results
def plot_results(x_test, x_sim):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True)

    axs[0, 0].plot(x_test[:, 0], x_test[:, 1], "k.", markersize=1)
    axs[0, 0].set(title="Stochastic system", ylabel="$\mu$", ylim=[4, 1])

    axs[1, 0].plot(x_test[:, 0], x_test[:, 1], "k.", markersize=1)
    axs[1, 0].set(ylabel="$\mu$", ylim=[4, 3.45])

    axs[0, 1].plot(x_sim[:, 0], x_sim[:, 1], "k.", markersize=1)
    axs[0, 1].set(title="Sparse identified system", ylabel="$\mu$", ylim=[4, 1])

    axs[1, 1].plot(x_sim[:, 0], x_sim[:, 1], "k.", markersize=1)
    axs[1, 1].set(ylabel="$\mu$", ylim=[4, 3.45])

    return fig


def example_article():
    N = 1000
    eps = 0.025
    x_train = generate_data(
        N, mus=[2.5, 2.75, 3, 3.25, 3.5, 3.75, 3.8, 3.85, 3.9, 3.95], eps=eps
    )
    model = fit_model(x_train)
    x_test = generate_test_data(N, mus=np.arange(1, 4, 0.1), eps=eps)
    x_sim = simulate_model(model, mus=np.arange(1, 4, 0.1), N=N)
    fig = plot_results(x_test, x_sim)
    plt.show()


def main():
    plt.plot(range(0, 500, 1), generate_data(N=500, mus=[1], eps=0)[0][:, 0])
    plt.show()


if __name__ == "__main__":
    main()
