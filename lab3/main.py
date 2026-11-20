# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.dif import DIF
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.utils.utility import standardizer
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


# %% Ex.1
def loda_scores(X_train, X_test, proj_vecs, n_bins):
    hist_range = (-5.0, 5.0)
    train_proj = X_train @ proj_vecs.T
    test_proj = X_train @ proj_vecs.T

    n_test = X_test.shape[0]
    num_proj = proj_vecs.shape[0]
    test_probs = np.zeros((n_test, num_proj))

    for i in range(num_proj):
        counts, bin_edges = np.histogram(
            train_proj[:, i], bins=n_bins, range=hist_range
        )
        probs = counts / len(X_train)
        bin_indices = np.digitize(test_proj[:, i], bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        test_probs[:, i] = probs[bin_indices]

    mean_probs = np.mean(test_probs, axis=1)
    return mean_probs


X_train, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
    n_samples=500, n_features=2, centers=[[0.0, 0.0]], cluster_std=1.0
)

raw_vecs = np.random.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2), size=5)
proj_vecs = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)

X_test = np.random.uniform(low=-3.0, high=3.0, size=(500, 2))

bin_values = [5, 10, 25, 50]
fig, axes = plt.subplots(1, 4, figsize=(20, 7), sharex=True, sharey=True)

for ax, n_bins in zip(axes, bin_values):
    scores = loda_scores(X_train, X_test, proj_vecs, n_bins)

    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis")
    ax.set_title(f"bins = {n_bins}")
    fig.colorbar(scatter, ax=ax, label="Mean Probability Score")

plt.suptitle("LODA variant")
plt.show()


# %% Ex.2
X_train, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
    n_samples=[500, 500],  # pyright: ignore[reportArgumentType]
    n_features=2,
    centers=[[10, 0], [0, 10]],
    cluster_std=1.0,
)
X_test = np.random.uniform(low=-10, high=20, size=(1000, 2))
contamination = 0.02

ifor = IForest(contamination=contamination)
dif = DIF(contamination=contamination)
loda = LODA(contamination=contamination)
models = [("IForest", ifor), ("DIF", dif), ("LODA", loda)]

fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)

for ax, (name, model) in zip(axes, models):
    model.fit(X_train)
    scores = model.decision_function(X_test)

    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis")

    ax.set_title(f"{name}")
    fig.colorbar(scatter, ax=ax, label="Anomaly Score")

plt.suptitle("Model Comparison (2D)")
plt.show()


# %% Ex.2.5
dif_neurons = [[16, 8], [64, 32], [256, 128]]
loda_bins = [5, 15, 50]

fig, axes = plt.subplots(2, 3, figsize=(20, 14), sharex=True, sharey=True)

for ax, neurons in zip(axes[0], dif_neurons):
    dif = DIF(hidden_neurons=neurons, contamination=contamination)
    dif.fit(X_train)
    scores = dif.decision_function(X_test)

    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis")
    ax.set_title(f"DIF (hidden_neurons={neurons})")
    fig.colorbar(scatter, ax=ax, label="Anomaly Score")

for ax, bins in zip(axes[1], loda_bins):
    loda = LODA(n_bins=bins, contamination=contamination)
    loda.fit(X_train)
    scores = loda.decision_function(X_test)

    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis")
    ax.set_title(f"LODA (n_bins={bins})")
    fig.colorbar(scatter, ax=ax, label="Anomaly Score")

plt.suptitle("DIF & LODA configurations")
plt.show()


# %% Ex.2.6
X_train, _ = make_blobs(  # pyright: ignore[reportAssignmentType]
    n_samples=[500, 500],  # pyright: ignore[reportArgumentType]
    n_features=3,
    centers=[[0, 10, 0], [10, 0, 10]],
    cluster_std=1.0,
)
X_test = np.random.uniform(low=-10, high=20, size=(1000, 3))
contamination = 0.02

ifor = IForest(contamination=contamination)
dif = DIF(contamination=contamination)
loda = LODA(contamination=contamination)
models = [("IForest", ifor), ("DIF", dif), ("LODA", loda)]

fig = plt.figure(figsize=(20, 7))

for i, (name, model) in enumerate(models, start=1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    model.fit(X_train)
    scores = model.decision_function(X_test)

    scatter = ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        X_test[:, 2],  # pyright: ignore[reportArgumentType]
        c=scores,
        cmap="viridis",
    )

    ax.set_title(f"{name}")
    fig.colorbar(scatter, ax=ax, label="Anomaly Score")

plt.suptitle("Model Comparison (3D)")
plt.show()


# %% Ex.3
data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

n_splits = 10
results = {
    "IForest": {"ba": [], "roc_auc": []},
    "DIF": {"ba": [], "roc_auc": []},
    "LODA": {"ba": [], "roc_auc": []},
}

for i_split in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)
    X_train, X_test = standardizer(X_train, X_test)  # pyright: ignore[reportAssignmentType]
    contamination = np.clip(np.mean(y_train), 1e-4, 0.5)

    ifor = IForest(contamination=contamination)
    dif = DIF(contamination=contamination)
    loda = LODA(contamination=contamination)
    models = [("IForest", ifor), ("DIF", dif), ("LODA", loda)]

    for name, model in models:
        model.fit(X_train)

        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)

        ba = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)

        results[name]["ba"].append(ba)
        results[name]["roc_auc"].append(roc_auc)
    print(f"Split {i_split + 1}/{n_splits}")

for name, stats in results.items():
    mean_ba = np.mean(stats["ba"])
    mean_auc = np.mean(stats["roc_auc"])
    print(f"Model: {name}\n\tMean BA: {mean_ba}\n\tMean ROC AUC: {mean_auc}\n")
