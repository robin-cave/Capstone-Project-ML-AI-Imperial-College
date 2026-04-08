"""
Surrogate models for Bayesian optimization.
Implements SurrogateModel ABC and GPSurrogate; add SVMSurrogate, MLPSurrogate here.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class SurrogateModel(ABC):
    """Abstract base class for surrogate models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) predictions."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class GPSurrogate(SurrogateModel):
    """Gaussian Process surrogate with RBF kernel. Supports ARD for per-dimension length scales."""

    def __init__(
        self,
        length_scale: float = 1.0,
        length_scale_bounds: Tuple[float, float] = (1e-2, 1e3),
        noise: float = 1e-5,
        optimize: bool = True,
        use_ard: bool = False,
    ):
        self.length_scale = length_scale
        # Upper bound 1e3 avoids ConvergenceWarning when a dimension has very long length scale
        self.length_scale_bounds = length_scale_bounds
        self.noise = noise
        self.optimize = optimize
        self.use_ard = use_ard
        if not use_ard:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, length_scale_bounds)
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=noise,
                n_restarts_optimizer=10 if optimize else 0,
                normalize_y=True,
                random_state=42,
            )
        else:
            self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.use_ard:
            n_dims = X.shape[1]
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=np.ones(n_dims) * self.length_scale,
                length_scale_bounds=self.length_scale_bounds,
            )
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise,
                n_restarts_optimizer=10 if self.optimize else 0,
                normalize_y=True,
                random_state=42,
            )
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        mean, std = self.model.predict(X, return_std=True)
        return mean, std

    def get_name(self) -> str:
        if self.use_ard and self.is_fitted:
            return "GP-RBF-ARD"
        return f"GP-RBF (ls={self.length_scale:.3f})"

    def get_length_scales(self) -> Optional[np.ndarray]:
        """Return learned length scales (per-dimension if use_ard=True). Only valid after fit."""
        if not self.is_fitted or self.model.kernel_ is None:
            return None
        k = self.model.kernel_
        if hasattr(k, "k2") and hasattr(k.k2, "length_scale"):
            return np.atleast_1d(k.k2.length_scale)
        return None


class SVMSurrogate(SurrogateModel):
    """SVR with RBF kernel; uncertainty via bootstrap ensemble of SVRs."""

    def __init__(
        self,
        C: float = 10.0,
        epsilon: float = 0.1,
        gamma: Union[str, float] = "scale",
        n_bootstrap: int = 20,
        random_state: int = 42,
    ):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.models: list = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        self.models = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_boot = X_scaled[idx]
            y_boot = y_scaled[idx]
            m = SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma=self.gamma)
            m.fit(X_boot, y_boot)
            self.models.append(m)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X = np.asarray(X)
        X_scaled = self.scaler_X.transform(X)
        preds = np.array([m.predict(X_scaled) for m in self.models])
        y_scaled_mean = preds.mean(axis=0)
        y_scaled_std = np.maximum(preds.std(axis=0), 1e-9)
        mean = self.scaler_y.inverse_transform(y_scaled_mean.reshape(-1, 1)).ravel()
        # Approximate std in original scale via scaling factor (scale is 1D)
        scale = self.scaler_y.scale_[0]
        std = y_scaled_std * scale
        return mean, std

    def get_name(self) -> str:
        return f"SVR-RBF (C={self.C}, n_boot={self.n_bootstrap})"


class MLPSurrogate(SurrogateModel):
    """PyTorch MLP surrogate; uncertainty via MC Dropout at prediction time."""

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 500,
        n_mc_samples: int = 50,
        random_state: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.n_mc_samples = n_mc_samples
        self.random_state = random_state
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self._model = None
        self._input_dim: Optional[int] = None
        self.is_fitted = False

    def _build_model(self, input_dim: int) -> "torch.nn.Module":
        import torch
        layers = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(self.dropout))
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self._input_dim = X.shape[1]
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        torch.manual_seed(self.random_state)
        self._model = self._build_model(self._input_dim)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_scaled, dtype=torch.float32).reshape(-1, 1)
        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self._model(X_t)
            loss = torch.nn.functional.mse_loss(pred, y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        import torch
        X = np.asarray(X)
        X_scaled = self.scaler_X.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        self._model.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                pred = self._model(X_t)
                preds.append(pred.numpy())
        preds = np.concatenate(preds, axis=1)
        y_scaled_mean = preds.mean(axis=1)
        y_scaled_std = np.maximum(preds.std(axis=1), 1e-9)
        mean = self.scaler_y.inverse_transform(y_scaled_mean.reshape(-1, 1)).ravel()
        scale = self.scaler_y.scale_[0]
        std = y_scaled_std * scale
        return mean, std

    def get_name(self) -> str:
        hs = "-".join(map(str, self.hidden_sizes))
        return f"MLP ({hs}, dropout={self.dropout})"


def gp_mean_gradient(
    surrogate: GPSurrogate,
    x: np.ndarray,
    eps: float = 1e-5,
    bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> np.ndarray:
    """
    Compute gradient of the GP posterior mean at x via central finite differences.
    """
    x = np.asarray(x).ravel()
    n = x.size
    grad = np.zeros(n)
    low, high = bounds[0], bounds[1]
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        x_plus = np.clip(x + eps * e, low, high - 1e-8)
        x_minus = np.clip(x - eps * e, low, high - 1e-8)
        mu_plus = surrogate.predict(x_plus.reshape(1, -1))[0][0]
        mu_minus = surrogate.predict(x_minus.reshape(1, -1))[0][0]
        grad[j] = (mu_plus - mu_minus) / (2.0 * eps)
    return grad
