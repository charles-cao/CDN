"""Cauchy Displacement Network (CDN) for tabular anomaly detection.

Training: each training point is regressed onto the mean of its k nearest
training neighbours by a single-hidden-layer network with Cauchy activations.
Scoring: s(x) = ||x - f(x)||, computed with one forward pass and no
neighbour search; the training data are not needed after fitting.
"""

import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pyod.models.base import BaseDetector


class CauchyActivation(nn.Module):
    """Cauchy activation with trainable parameters for every hidden unit.

    phi_j(z) = (alpha_j * z + beta_j) / (z^2 + gamma_j^2 + eps)
    """

    def __init__(self, num_features, alpha_init=0.1, beta_init=0.2,
                 gamma_init=0.6, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_features,), alpha_init))
        self.beta = nn.Parameter(torch.full((num_features,), beta_init))
        self.gamma = nn.Parameter(torch.full((num_features,), gamma_init))
        self.eps = eps

    def forward(self, z):
        return (self.alpha * z + self.beta) / (
            z.square() + self.gamma.square() + self.eps)


class _Network(nn.Module):
    """f(x) = W2 phi(W1 x + b1) + b2."""

    def __init__(self, d, hidden, alpha_init, beta_init, gamma_init):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act = CauchyActivation(hidden, alpha_init, beta_init, gamma_init)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CDN(BaseDetector):
    """Cauchy Displacement Network.

    The defaults are the configuration used for all datasets in the paper.

    Parameters
    ----------
    hidden : int, hidden width h
    k : int, number of neighbours used to build the training targets
    lr : float, Adam learning rate
    batch_size : int, mini-batch size (min(batch_size, n) is used)
    steps : int, number of training steps T
    wd : float, Adam weight decay
    alpha_init, beta_init, gamma_init : float, initial alpha_j, beta_j,
        gamma_j of every Cauchy unit
    contamination : float, expected proportion of anomalies, used only by
        PyOD to set the threshold of predict()
    random_state : int, seed for initialisation and mini-batch sampling
    """

    def __init__(self, hidden=128, k=5, lr=1e-3, batch_size=32, steps=2000,
                 wd=1e-5, alpha_init=0.1, beta_init=0.2, gamma_init=0.6,
                 contamination=0.1, random_state=42):
        super().__init__(contamination=contamination)
        self.hidden, self.k, self.lr = hidden, k, lr
        self.batch_size, self.steps, self.wd = batch_size, steps, wd
        self.alpha_init, self.beta_init = alpha_init, beta_init
        self.gamma_init = gamma_init
        self.random_state = random_state

    def _train(self, Xs, target):
        torch.manual_seed(self.random_state)
        n, d = Xs.shape
        self.net_ = _Network(d, self.hidden, self.alpha_init,
                             self.beta_init, self.gamma_init)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        X, T = torch.tensor(Xs), torch.tensor(target)
        bs = min(self.batch_size, n)
        g = torch.Generator().manual_seed(self.random_state)
        self.net_.train()
        for _ in range(self.steps):
            i = torch.randint(n, (bs,), generator=g)
            opt.zero_grad()
            loss = ((self.net_(X[i]) - T[i]) ** 2).mean()
            loss.backward()
            opt.step()
        self.net_.eval()

    def fit(self, X, y=None):
        """Fit on normal training data X of shape (n, d)."""
        self._set_n_classes(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype('float32')
        # targets: mean of the k nearest other training points
        knn = NearestNeighbors(n_neighbors=min(self.k + 1, len(Xs))).fit(Xs)
        idx = knn.kneighbors(Xs, return_distance=False)[:, 1:]
        target = Xs[idx].mean(1).astype('float32')
        self._train(Xs, target)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Anomaly scores of X; larger means more anomalous."""
        Xs = torch.tensor(self.scaler_.transform(X).astype('float32'))
        with torch.no_grad():
            return (self.net_(Xs) - Xs).norm(dim=1).numpy()
