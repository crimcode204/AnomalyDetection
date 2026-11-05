# %% Imports
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from pyod.models.combination import average, maximization
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.data import generate_data_clusters
from pyod.utils.utility import standardizer
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


# %% Ex. 1 data
@dataclass
class Scenario:
    title: str
    out_x_mu: float
    out_x_sig: float
    out_y_mu: float
    out_y_sig: float


def calc_leverage_scores(X):
    XTX_inv = np.linalg.pinv(X.T @ X)
    leverage_scores = np.sum((X @ XTX_inv) * X, axis=1)
    return leverage_scores


n_reg = 90
n_out = 10

x_mu, x_sig, y_sig = 0.0, 1.0, 0.5
scenarios = [
    Scenario("Regular", 0, 1, 0, 0.5),
    Scenario("High X variance", 10, 2, 0, 0.5),
    Scenario("High Y variance", 0, 1, 15, 5),
    Scenario("High X&Y variance", 10, 2, 15, 5),
]

# %% Ex. 1 1D
a, b = 5.0, 7.0
fig, axes = plt.subplots(2, 2)

X_reg = np.random.normal(x_mu, x_sig, n_reg)
y_reg = a * X_reg + b + np.random.normal(0, y_sig, n_reg)

for ax, sc in zip(axes.flatten(), scenarios):
    X_out = np.random.normal(sc.out_x_mu, sc.out_x_sig, n_out)
    y_out = a * X_out + b + np.random.normal(sc.out_y_mu, sc.out_y_sig, n_out)

    X, y = np.concat((X_reg, X_out)), np.concat((y_reg, y_out))

    design_matrix = np.column_stack((X, np.ones_like(X)))
    scores = calc_leverage_scores(design_matrix)
    top_score_points = np.argsort(scores)[-n_out:]

    scatter = ax.scatter(X, y, c=scores, cmap="coolwarm")
    ax.scatter(
        X[top_score_points],
        y[top_score_points],
        edgecolors="red",
        facecolors="none",
    )
    ax.set_title(sc.title)
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

# %% Ex. 1 2D
a, b, c = 2.0, -3.0, 0.5

X1_reg = np.random.normal(x_mu, x_sig, n_reg)
X2_reg = np.random.normal(x_mu, x_sig, n_reg)
y_reg = a * X1_reg + b * X2_reg + c + np.random.normal(0, y_sig, n_reg)

fig = plt.figure(figsize=(12, 10))

for i, sc in enumerate(scenarios):
    X1_out = np.random.normal(sc.out_x_mu, sc.out_x_sig, n_out)
    X2_out = np.random.normal(-sc.out_x_mu, sc.out_x_sig, n_out)
    y_out = (
        a * X1_out + b * X2_out + c + np.random.normal(sc.out_y_mu, sc.out_y_sig, n_out)
    )

    X1, X2, y = (
        np.concat((X1_reg, X1_out)),
        np.concat((X2_reg, X2_out)),
        np.concat((y_reg, y_out)),
    )

    design_matrix = np.column_stack((X1, X2, np.ones_like(X1)))
    scores = calc_leverage_scores(design_matrix)
    top_score_points = np.argsort(scores)[-n_out:]

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    scatter = ax.scatter(X1, y, X2, c=scores)  # pyright: ignore[reportArgumentType]
    ax.scatter(
        X1[top_score_points],
        y[top_score_points],
        X2[top_score_points],  # pyright: ignore[reportArgumentType]
        edgecolors="red",
        facecolors="none",
        s=50,
    )
    ax.set_title(sc.title)

plt.tight_layout()
plt.show()

# %% Ex. 2

X_train, X_test, y_train, y_test = generate_data_clusters(  # pyright: ignore[reportAssignmentType]
    n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1
)

neighbors = [5, 10, 15]

for n_neighbors in neighbors:
    knn = KNN(n_neighbors=n_neighbors)
    knn.fit(X_train)

    y_train_pred = knn.labels_
    y_test_pred = knn.predict(X_test)

    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f"Balanced Accuracy (n_neighbors={n_neighbors}):")
    print(f"train: {ba_train}\ntest: {ba_test}")

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm")  # pyright: ignore[reportArgumentType, reportCallIssue]
    axes[0, 0].set_title("Ground Truth - Train Data")

    axes[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap="coolwarm")  # pyright: ignore[reportArgumentType, reportCallIssue]
    axes[0, 1].set_title("Predicted - Train Data")

    axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm")  # pyright: ignore[reportArgumentType, reportCallIssue]
    axes[1, 0].set_title("Ground Truth - Test Data")

    axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap="coolwarm")  # pyright: ignore[reportArgumentType, reportCallIssue]
    axes[1, 1].set_title("Predicted - Test Data")

    plt.tight_layout()
    plt.show()

# %% Ex. 3

n_samples = [200, 100]
centers = [[-10, -10], [10, 10]]
cluster_std = [2.0, 6.0]

X, y = make_blobs(  # pyright: ignore[reportAssignmentType]
    n_samples=n_samples,  # pyright: ignore[reportArgumentType]
    n_features=2,
    centers=centers,
    cluster_std=cluster_std,  # pyright: ignore[reportArgumentType]
)

contamination = 0.07
neighbors = [10, 20, 50]

for n_neighbors in neighbors:
    knn = KNN(n_neighbors=n_neighbors, contamination=contamination)
    knn.fit(X)
    knn_pred = knn.labels_

    lof = LOF(n_neighbors=n_neighbors, contamination=contamination)
    lof.fit(X)
    lof_pred = lof.labels_

    fig, axes = plt.subplots(1, 2)

    print(f"n_neighbors={n_neighbors}")

    axes[0].scatter(X[:, 0], X[:, 1], c=knn_pred, cmap="coolwarm")  # pyright: ignore[reportArgumentType, reportCallIssue]
    axes[0].set_title("KNN")

    axes[1].scatter(X[:, 0], X[:, 1], c=lof_pred, cmap="coolwarm")  # pyright: ignore[reportArgumentType, reportCallIssue]
    axes[1].set_title("LOF")

    plt.tight_layout()
    plt.show()

# %% Ex. 4

mat = loadmat("cardio.mat")
X, y = mat["X"], mat["y"].ravel()

contamination = np.count_nonzero(y) / len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_test = standardizer(X_train, X_test)  # pyright: ignore[reportAssignmentType]

n_models = 10
neighbors = np.linspace(30, 120, n_models, dtype=int)

train_scores = np.zeros((X_train.shape[0], n_models))  # pyright: ignore[reportAttributeAccessIssue]
test_scores = np.zeros((X_test.shape[0], n_models))  # pyright: ignore[reportAttributeAccessIssue]

for i, n_neighbors in enumerate(neighbors):
    knn = KNN(n_neighbors=n_neighbors)
    knn.fit(X_train)

    train_scores[:, i] = knn.decision_scores_
    test_scores[:, i] = knn.decision_function(X_test)

    y_train_pred = knn.labels_
    y_test_pred = knn.predict(X_test)

    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)

    print(f"Balanced Accuracy (n_neighbors={n_neighbors}):")
    print(f"train: {ba_train}\ntest: {ba_test}\n")

train_scores, test_scores = standardizer(train_scores, test_scores)  # pyright: ignore[reportAssignmentType]

train_scores_avg = average(train_scores)
test_scores_avg = average(test_scores)
threshold_avg = np.quantile(train_scores_avg, 1 - contamination)

y_test_pred_avg = (test_scores_avg >= threshold_avg).astype(int)
ba_avg = balanced_accuracy_score(y_test, y_test_pred_avg)

train_scores_max = maximization(train_scores)
test_scores_max = maximization(test_scores)
threshold_max = np.quantile(train_scores_max, 1 - contamination)

y_test_pred_max = (test_scores_max >= threshold_avg).astype(int)
ba_max = balanced_accuracy_score(y_test, y_test_pred_max)

print(f"\nAverage strategy:\nThreshold: {threshold_avg}\nBalanced Accuracy: {ba_avg}\n")
print(f"Maximization strategy:\nThreshold: {threshold_max}\nBalanced Accuracy: {ba_max}\n")
