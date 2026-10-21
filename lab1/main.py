# %% Imports
import numpy as np
from matplotlib import pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, roc_curve

# %% Data gen.
(x_train, x_test, y_train, y_test) = generate_data(  # pyright: ignore[reportAssignmentType]
    n_train=400, n_test=100, n_features=2
)

# %% Ex. 1
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.title("Ex. 1")
plt.show()

# %% Ex. 2
knn = KNN(contamination=0.2)
knn.fit(x_train)

pred_train = knn.predict(x_train)
pred_test = knn.predict(x_test)
pred = np.concat((pred_train, pred_test))
truth = np.concat((y_train, y_test))

(tn, fp, fn, tp) = confusion_matrix(truth, pred).ravel().tolist()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print("Balanced accuracy: ", ba)

scores_train = knn.decision_function(x_train)
scores_test = knn.decision_function(x_test)
scores = np.concat((scores_train, scores_test))

(fpr, tpr, threshholds) = roc_curve(truth, scores)
plt.plot(fpr, tpr)
plt.show()
# When changing the contamination, the balanced accuracy changes, but the curve doesn't

# %% Ex. 3
contamination: float = 0.1
(x_train, x_test, y_train, y_test) = generate_data(  # pyright: ignore[reportAssignmentType]
    n_train=1000, n_test=0, n_features=1, contamination=contamination
)
x_train = x_train.flatten()

z_scores = np.abs((x_train - np.mean(x_train)) / np.std(x_train))
threshold = np.quantile(z_scores, 1 - contamination)
print("Z-score threshold: ", threshold)

pred = (z_scores >= threshold).astype(int)
truth = y_train

(tn, fp, fn, tp) = confusion_matrix(truth, pred).ravel().tolist()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print("Balanced accuracy: ", ba)

# %% Ex. 4
n_train = 1000
contamination = 0.1
n_outliers = int(n_train * contamination)
n_inliers = int(n_train * (1 - contamination))

mu = np.array([2.0, 3.0])
sigma = np.array([[4.0, 0.5], [0.5, 5.0]])

l = np.linalg.cholesky(sigma)

x_inliers = np.random.randn(n_inliers, 2)
y_inliers = x_inliers @ l.T + mu
x_outliers = np.random.randn(n_outliers, 2) * 3
y_outliers = x_outliers @ l.T + mu

y = np.vstack((y_inliers, y_outliers))
truth = np.concat((np.zeros(n_inliers), np.ones(n_outliers)))

emp_mu = np.mean(y, axis=0)
emp_sigma = np.cov(y, rowvar=False)
emp_l = np.linalg.cholesky(emp_sigma)

y_centered = y - emp_mu
x_reconstructed = y_centered @ np.linalg.inv(emp_l.T)
z_scores = np.linalg.norm(x_reconstructed, axis=1)

threshold = np.quantile(z_scores, 1 - contamination)
print("Z-score threshold: ", threshold)

pred = (z_scores >= threshold).astype(int)

(tn, fp, fn, tp) = confusion_matrix(truth, pred).ravel().tolist()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print("Balanced accuracy: ", ba)
