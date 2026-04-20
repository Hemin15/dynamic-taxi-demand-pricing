# model.py

import os
import io
import json
import copy
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# Optional model libraries
# -------------------------------------------------------------------
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor  # type: ignore
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

TARGET_COL = "demand"


BASE_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "pickup_lat_bin",
    "pickup_lon_bin",
    "zone_trip_volume",
    "date_ordinal",
]

LAG_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_24",
    "rolling_std_24",
    "expanding_mean_zone",
]

ALL_FEATURES = BASE_FEATURES + LAG_FEATURES


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, default=_json_default)


def save_dataframe(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def safe_smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true) + np.abs(y_pred), 1e-8, None)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
def add_temporal_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pickup_date"] = pd.to_datetime(out["pickup_date"])
    out["date_ordinal"] = out["pickup_date"].map(pd.Timestamp.toordinal)

    out = out.sort_values(["pickup_zone", "pickup_date", "hour"]).reset_index(drop=True)

    g = out.groupby("pickup_zone")["demand"]

    out["lag_1"] = g.transform(lambda s: s.shift(1))
    out["lag_2"] = g.transform(lambda s: s.shift(2))
    out["lag_24"] = g.transform(lambda s: s.shift(24))

    out["rolling_mean_3"] = g.transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    out["rolling_mean_24"] = g.transform(lambda s: s.shift(1).rolling(window=24, min_periods=1).mean())
    out["rolling_std_24"] = g.transform(lambda s: s.shift(1).rolling(window=24, min_periods=2).std())

    out["expanding_mean_zone"] = g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    return out


def prepare_model_data(demand_df: pd.DataFrame) -> pd.DataFrame:
    df = demand_df.copy()
    df["pickup_date"] = pd.to_datetime(df["pickup_date"], errors="coerce")
    df = df.dropna(subset=["pickup_date"]).copy()
    df = add_temporal_lag_features(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    return df


def time_based_split(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    unique_dates = pd.Index(sorted(df["pickup_date"].dropna().unique()))
    n = len(unique_dates)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = df[df["pickup_date"].isin(train_dates)].copy()
    val_df = df[df["pickup_date"].isin(val_dates)].copy()
    test_df = df[df["pickup_date"].isin(test_dates)].copy()

    return train_df, val_df, test_df


def get_xy(df: pd.DataFrame):
    return df[ALL_FEATURES].copy(), df[TARGET_COL].copy()


# -------------------------------------------------------------------
# Classical model factories
# -------------------------------------------------------------------
def create_tree_models(profile="final"):
    fast = profile == "cv"

    models = {
        "linear_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression())
        ]),
        "ridge_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=1.0))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=120 if fast else 180,
                max_depth=16 if fast else 20,
                min_samples_leaf=5 if fast else 4,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=160 if fast else 220,
                max_depth=None,
                min_samples_leaf=3 if fast else 2,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=150 if fast else 250,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ))
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=220 if fast else 320,
                learning_rate=0.05,
                max_depth=8,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42
            ))
        ]),
    }

    if HAS_XGB:
        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=220 if fast else 350,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
                eval_metric="rmse",
            ))
        ])

    if HAS_LGBM:
        models["lightgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=250 if fast else 400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            ))
        ])

    if HAS_CATBOOST:
        models["catboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostRegressor(
                iterations=220 if fast else 400,
                learning_rate=0.05,
                depth=8,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
                od_type="Iter",
                od_wait=25 if fast else 40
            ))
        ])

    return models


# -------------------------------------------------------------------
# Deep neural network for tabular regression
# -------------------------------------------------------------------
if HAS_TORCH:

    class TabularDataset(Dataset):
        def __init__(self, X, y=None, augment=False, noise_std=0.0):
            self.X = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
            self.y = None if y is None else torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)
            self.augment = augment
            self.noise_std = float(noise_std)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            x = self.X[idx]
            if self.augment and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.y is None:
                return x
            return x, self.y[idx]


    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.25):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.norm1 = nn.LayerNorm(dim)
            self.act = nn.SiLU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x):
            residual = x
            out = self.fc1(x)
            out = self.norm1(out)
            out = self.act(out)
            out = self.drop(out)
            out = self.fc2(out)
            out = self.norm2(out)
            out = out + residual
            out = self.act(out)
            return out


    class TabularRegressorNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, n_blocks=4, dropout=0.25):
            super().__init__()
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.act = nn.SiLU()
            self.drop = nn.Dropout(dropout)
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)
            ])
            mid_dim = max(hidden_dim // 2, 64)
            tail_dim = max(hidden_dim // 4, 32)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, mid_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(mid_dim, tail_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(tail_dim, 1),
            )

        def forward(self, x):
            x = self.input_layer(x)
            x = self.input_norm(x)
            x = self.act(x)
            x = self.drop(x)
            for block in self.blocks:
                x = block(x)
            x = self.head(x).squeeze(-1)
            return x


    class DeepTabularRegressor:
        """
        Deep residual MLP for tabular regression.

        - Standardizes inputs
        - Learns on log1p(target) with standardization
        - Uses dropout + residual blocks + weight decay
        - Uses early stopping and LR scheduling
        - Supports pickling via __getstate__/__setstate__
        """

        def __init__(
            self,
            hidden_dim=256,
            n_blocks=4,
            dropout=0.25,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=2048,
            max_epochs=35,
            patience=8,
            min_delta=1e-4,
            noise_std=0.01,
            device_name=None,
        ):
            self.hidden_dim = int(hidden_dim)
            self.n_blocks = int(n_blocks)
            self.dropout = float(dropout)
            self.lr = float(lr)
            self.weight_decay = float(weight_decay)
            self.batch_size = int(batch_size)
            self.max_epochs = int(max_epochs)
            self.patience = int(patience)
            self.min_delta = float(min_delta)
            self.noise_std = float(noise_std)
            self.device_name = device_name or ("cuda" if torch.cuda.is_available() else "cpu")

            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()

            self.net = None
            self.input_dim = None
            self.history_ = []
            self.best_epoch_ = None
            self.best_val_rmse_ = None
            self.best_val_mae_ = None
            self._fitted = False

        def _build_network(self, input_dim):
            return TabularRegressorNet(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
            )

        def _prepare_features(self, X):
            return np.asarray(X, dtype=np.float32)

        def _prepare_target(self, y):
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            y = np.log1p(np.clip(y, a_min=0.0, a_max=None))
            return y

        def _inverse_target(self, y_scaled):
            y_scaled = np.asarray(y_scaled, dtype=np.float32).reshape(-1, 1)
            y_log = self.target_scaler.inverse_transform(y_scaled).ravel()
            y = np.expm1(y_log)
            return np.clip(y, 0.0, None)

        def _predict_scaled_target(self, X):
            if self.net is None:
                raise RuntimeError("Neural network is not fitted yet.")

            self.net.eval()
            Xn = self.feature_scaler.transform(self._prepare_features(X))
            X_tensor = torch.tensor(Xn, dtype=torch.float32, device=self.device_name)

            with torch.no_grad():
                preds_scaled = self.net(X_tensor).detach().cpu().numpy()

            return preds_scaled

        def predict(self, X):
            preds_scaled = self._predict_scaled_target(X)
            return self._inverse_target(preds_scaled)

        def _evaluate_arrays(self, X, y):
            y_true = np.asarray(y, dtype=np.float32)
            y_pred = self.predict(X)

            return {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
                "median_ae": float(median_absolute_error(y_true, y_pred)),
                "mape": safe_mape(y_true, y_pred),
                "smape": safe_smape(y_true, y_pred),
                "bias": float(np.mean(y_pred - y_true)),
            }

        def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
            if not HAS_TORCH:
                raise RuntimeError("PyTorch is not available, so the neural network branch cannot be trained.")

            X_train = self._prepare_features(X_train)
            y_train = np.asarray(y_train, dtype=np.float32)

            if X_val is None or y_val is None:
                split_idx = int(len(X_train) * 0.9)
                X_train, X_val = X_train[:split_idx], X_train[split_idx:]
                y_train, y_val = y_train[:split_idx], y_train[split_idx:]

            X_val = self._prepare_features(X_val)
            y_val = np.asarray(y_val, dtype=np.float32)

            self.input_dim = X_train.shape[1]

            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)

            y_train_log = self._prepare_target(y_train)
            y_val_log = self._prepare_target(y_val)

            y_train_scaled = self.target_scaler.fit_transform(y_train_log).ravel().astype(np.float32)
            y_val_scaled = self.target_scaler.transform(y_val_log).ravel().astype(np.float32)

            train_ds = TabularDataset(
                X_train_scaled,
                y_train_scaled,
                augment=True,
                noise_std=self.noise_std,
            )
            val_ds = TabularDataset(X_val_scaled, y_val_scaled, augment=False, noise_std=0.0)

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

            self.net = self._build_network(self.input_dim).to(self.device_name)
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-5
            )
            criterion = nn.SmoothL1Loss()

            best_state = None
            best_val_rmse = float("inf")
            best_val_mae = float("inf")
            patience_counter = 0
            self.history_ = []

            for epoch in range(1, self.max_epochs + 1):
                self.net.train()
                train_losses = []

                for xb, yb in train_loader:
                    xb = xb.to(self.device_name)
                    yb = yb.to(self.device_name)

                    optimizer.zero_grad(set_to_none=True)
                    preds = self.net(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_losses.append(float(loss.detach().cpu().item()))

                train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

                # Validation prediction on original scale
                val_metrics = self._evaluate_arrays(X_val, y_val)
                scheduler.step(val_metrics["rmse"])

                current_lr = optimizer.param_groups[0]["lr"]
                epoch_row = {
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_median_ae": val_metrics["median_ae"],
                    "val_mape": val_metrics["mape"],
                    "val_smape": val_metrics["smape"],
                    "val_bias": val_metrics["bias"],
                    "lr": float(current_lr),
                }
                self.history_.append(epoch_row)

                if verbose:
                    print(
                        f"[NN] epoch {epoch:03d} | "
                        f"train_loss={train_loss:.5f} | "
                        f"val_mae={val_metrics['mae']:.4f} | "
                        f"val_rmse={val_metrics['rmse']:.4f} | "
                        f"lr={current_lr:.6f}"
                    )

                improved = val_metrics["rmse"] < (best_val_rmse - self.min_delta)
                if improved:
                    best_val_rmse = val_metrics["rmse"]
                    best_val_mae = val_metrics["mae"]
                    best_state = copy.deepcopy(self.net.state_dict())
                    self.best_epoch_ = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"[NN] Early stopping at epoch {epoch}")
                    break

            if best_state is not None:
                self.net.load_state_dict(best_state)

            self.best_val_rmse_ = float(best_val_rmse)
            self.best_val_mae_ = float(best_val_mae)
            self._fitted = True
            return self

        def training_history_dataframe(self):
            return pd.DataFrame(self.history_)

        def __getstate__(self):
            state = self.__dict__.copy()
            net_bytes = None
            if self.net is not None and HAS_TORCH:
                buffer = io.BytesIO()
                torch.save(self.net.state_dict(), buffer)
                net_bytes = buffer.getvalue()

            state["_net_state_bytes"] = net_bytes
            state["net"] = None
            return state

        def __setstate__(self, state):
            net_bytes = state.pop("_net_state_bytes", None)
            self.__dict__.update(state)

            if net_bytes is not None and HAS_TORCH and self.input_dim is not None:
                self.net = self._build_network(self.input_dim).to(self.device_name)
                buffer = io.BytesIO(net_bytes)
                self.net.load_state_dict(torch.load(buffer, map_location=self.device_name))
                self.net.eval()
            else:
                self.net = None


else:
    class DeepTabularRegressor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available in this environment.")


# -------------------------------------------------------------------
# Model selection helpers
# -------------------------------------------------------------------
def candidate_model_names(include_nn=True):
    names = [
        "linear_regression",
        "ridge_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
    ]
    if HAS_XGB:
        names.append("xgboost")
    if HAS_LGBM:
        names.append("lightgbm")
    if HAS_CATBOOST:
        names.append("catboost")
    if include_nn and HAS_TORCH:
        names.append("neural_network")
    return names


def create_model(name: str, profile="final"):
    """
    profile:
        - "cv"    -> faster configurations for walk-forward cross-validation
        - "final" -> stronger configurations for final holdout training
    """
    fast = profile == "cv"

    if name == "linear_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression())
        ])

    if name == "ridge_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=1.0))
        ])

    if name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=120 if fast else 180,
                max_depth=16 if fast else 20,
                min_samples_leaf=5 if fast else 4,
                random_state=42,
                n_jobs=-1
            ))
        ])

    if name == "extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=160 if fast else 220,
                max_depth=None,
                min_samples_leaf=3 if fast else 2,
                random_state=42,
                n_jobs=-1
            ))
        ])

    if name == "gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=150 if fast else 250,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ))
        ])

    if name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=220 if fast else 320,
                learning_rate=0.05,
                max_depth=8,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42
            ))
        ])

    if name == "xgboost" and HAS_XGB:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=220 if fast else 350,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
                eval_metric="rmse",
            ))
        ])

    if name == "lightgbm" and HAS_LGBM:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=250 if fast else 400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            ))
        ])

    if name == "catboost" and HAS_CATBOOST:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostRegressor(
                iterations=220 if fast else 400,
                learning_rate=0.05,
                depth=8,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
                od_type="Iter",
                od_wait=25 if fast else 40
            ))
        ])

    if name == "neural_network" and HAS_TORCH:
        return DeepTabularRegressor(
            hidden_dim=192 if fast else 256,
            n_blocks=3 if fast else 4,
            dropout=0.30 if fast else 0.25,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=4096 if fast else 2048,
            max_epochs=12 if fast else 35,
            patience=5 if fast else 8,
            min_delta=1e-4,
            noise_std=0.015,
            device_name="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        )

    raise ValueError(f"Unknown or unavailable model name: {name}")


# -------------------------------------------------------------------
# Evaluation and diagnostics
# -------------------------------------------------------------------
def evaluate_model(model, X, y, label="set"):
    preds = model.predict(X)
    y_true = np.asarray(y)
    y_pred = np.asarray(preds)

    metrics = {
        "label": label,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "mape": safe_mape(y_true, y_pred),
        "smape": safe_smape(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }
    return metrics, preds


def make_prediction_frame(df, y_true, y_pred):
    out = df[[
        "pickup_date",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_peak_hour",
        "pickup_zone"
    ]].copy()

    out["actual_demand"] = np.asarray(y_true)
    out["predicted_demand"] = np.asarray(y_pred)
    out["error"] = out["actual_demand"] - out["predicted_demand"]
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = out["error"] ** 2
    out["pct_error"] = np.where(
        out["actual_demand"] != 0,
        out["abs_error"] / out["actual_demand"],
        np.nan
    )
    out["over_prediction"] = (out["predicted_demand"] > out["actual_demand"]).astype(int)
    out["under_prediction"] = (out["predicted_demand"] < out["actual_demand"]).astype(int)
    return out


def summarize_errors(pred_df: pd.DataFrame):
    hour_summary = (
        pred_df.groupby("hour", as_index=False, observed=False)
        .agg(
            count=("abs_error", "size"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_error=("error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            p95_abs_error=("abs_error", lambda s: float(np.percentile(s, 95)))
        )
        .sort_values("hour")
    )

    zone_summary = (
        pred_df.groupby("pickup_zone", as_index=False, observed=False)
        .agg(
            count=("abs_error", "size"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_error=("error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            p95_abs_error=("abs_error", lambda s: float(np.percentile(s, 95)))
        )
        .sort_values("mean_abs_error", ascending=False)
    )

    peak_summary = (
        pred_df.groupby("is_peak_hour", as_index=False, observed=False)
        .agg(
            count=("abs_error", "size"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_error=("error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            p95_abs_error=("abs_error", lambda s: float(np.percentile(s, 95)))
        )
        .sort_values("is_peak_hour")
    )

    return hour_summary, zone_summary, peak_summary


def calibration_summary(pred_df: pd.DataFrame, n_bins=10):
    temp = pred_df[["actual_demand", "predicted_demand"]].copy()
    temp["bin"] = pd.qcut(temp["predicted_demand"], q=n_bins, duplicates="drop")

    cal = (
        temp.groupby("bin", as_index=False, observed=False)
        .agg(
            mean_predicted=("predicted_demand", "mean"),
            mean_actual=("actual_demand", "mean"),
            count=("actual_demand", "size")
        )
    )
    cal["calibration_gap"] = cal["mean_actual"] - cal["mean_predicted"]
    return cal


def residual_summary(pred_df: pd.DataFrame):
    err = pred_df["error"].to_numpy()
    abs_err = pred_df["abs_error"].to_numpy()

    return {
        "count": int(len(pred_df)),
        "mean_error": float(np.mean(err)),
        "std_error": float(np.std(err)),
        "median_error": float(np.median(err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "rmse": float(np.sqrt(np.mean(pred_df["sq_error"]))),
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
    }


def extract_feature_importance(model, feature_names):
    """
    Native feature importance for tree models and coefficients for linear models.
    Neural networks will fall back to permutation importance.
    """
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": estimator.feature_importances_
            }).sort_values("importance", ascending=False)
            return fi

        if hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_).ravel()
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": np.abs(coef),
                "raw_coefficient": coef
            }).sort_values("importance", ascending=False)
            return fi

    return None


def compute_permutation_importance(model, X_sample, y_sample, feature_names, n_repeats=5):
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="neg_mean_absolute_error"
    )

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    return perm_df


# -------------------------------------------------------------------
# Cross-validation
# -------------------------------------------------------------------
def run_walk_forward_cv(df: pd.DataFrame, model_names=None, n_splits=3, profile="cv"):
    """
    Time-series cross-validation on chronological date splits.
    Uses fold-by-fold training and validation.
    """
    if model_names is None:
        model_names = candidate_model_names(include_nn=True)

    unique_dates = pd.Index(sorted(df["pickup_date"].dropna().unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    rows = []

    print("\n=== WALK-FORWARD CROSS-VALIDATION ===")
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = unique_dates[train_idx]
        val_dates = unique_dates[val_idx]

        fold_train_df = df[df["pickup_date"].isin(train_dates)].copy()
        fold_val_df = df[df["pickup_date"].isin(val_dates)].copy()

        X_train, y_train = get_xy(fold_train_df)
        X_val, y_val = get_xy(fold_val_df)

        print(f"\nFold {fold_idx}: train_rows={len(fold_train_df)}, val_rows={len(fold_val_df)}")

        for name in model_names:
            model = create_model(name, profile=profile)

            if name == "neural_network":
                model.fit(X_train, y_train, X_val, y_val, verbose=False)
            else:
                model.fit(X_train, y_train)

            train_metrics, _ = evaluate_model(model, X_train, y_train, label="cv_train")
            val_metrics, _ = evaluate_model(model, X_val, y_val, label="cv_val")

            rows.append({
                "fold": fold_idx,
                "model_name": name,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "mae_gap": val_metrics["mae"] - train_metrics["mae"],
                "rmse_gap": val_metrics["rmse"] - train_metrics["rmse"],
                "bias": val_metrics["bias"],
                "median_ae": val_metrics["median_ae"],
                "mape": val_metrics["mape"],
                "smape": val_metrics["smape"],
            })

            print(
                f"  {name:22s} | "
                f"train_mae={train_metrics['mae']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f}"
            )

    cv_results_df = pd.DataFrame(rows)

    cv_summary_df = (
        cv_results_df.groupby("model_name", as_index=False, observed=False)
        .agg(
            cv_train_mae_mean=("train_mae", "mean"),
            cv_train_mae_std=("train_mae", "std"),
            cv_val_mae_mean=("val_mae", "mean"),
            cv_val_mae_std=("val_mae", "std"),
            cv_train_rmse_mean=("train_rmse", "mean"),
            cv_val_rmse_mean=("val_rmse", "mean"),
            cv_val_r2_mean=("val_r2", "mean"),
            cv_gap_mae_mean=("mae_gap", "mean"),
            cv_gap_rmse_mean=("rmse_gap", "mean"),
            cv_bias_mean=("bias", "mean"),
            cv_median_ae_mean=("median_ae", "mean"),
            cv_mape_mean=("mape", "mean"),
            cv_smape_mean=("smape", "mean"),
        )
        .sort_values("cv_val_mae_mean")
        .reset_index(drop=True)
    )

    return cv_results_df, cv_summary_df


# -------------------------------------------------------------------
# Main training routine
# -------------------------------------------------------------------
def train_model(demand_df: pd.DataFrame):
    set_seed(42)

    df = prepare_model_data(demand_df)
    df = df.dropna(subset=ALL_FEATURES + [TARGET_COL]).copy()

    train_df, val_df, test_df = time_based_split(df)

    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(val_df)
    X_test, y_test = get_xy(test_df)

    candidate_names = candidate_model_names(include_nn=True)

    # ----------------------------------------------------------------
    # Cross-validation on the training segment only
    # ----------------------------------------------------------------
    cv_results_df, cv_summary_df = run_walk_forward_cv(
        train_df,
        model_names=candidate_names,
        n_splits=3,
        profile="cv"
    )

    save_dataframe(cv_results_df, os.path.join(ARTIFACTS_DIR, "cross_validation_results.csv"))
    save_dataframe(cv_summary_df, os.path.join(ARTIFACTS_DIR, "cross_validation_summary.csv"))

    # ----------------------------------------------------------------
    # Final holdout training and evaluation
    # ----------------------------------------------------------------
    comparison_rows = []
    fitted_models = {}

    print("\n=== MODEL COMPARISON ON HOLDOUT SPLIT ===")
    for name in candidate_names:
        model = create_model(name, profile="final")

        if name == "neural_network":
            model.fit(X_train, y_train, X_val, y_val, verbose=True)
        else:
            model.fit(X_train, y_train)

        fitted_models[name] = model

        train_metrics, train_preds = evaluate_model(model, X_train, y_train, label="train")
        val_metrics, val_preds = evaluate_model(model, X_val, y_val, label="validation")
        test_metrics, test_preds = evaluate_model(model, X_test, y_test, label="test")

        comparison_rows.append({
            "model_name": name,
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "train_median_ae": train_metrics["median_ae"],
            "val_median_ae": val_metrics["median_ae"],
            "test_median_ae": test_metrics["median_ae"],
            "train_mape": train_metrics["mape"],
            "val_mape": val_metrics["mape"],
            "test_mape": test_metrics["mape"],
            "train_smape": train_metrics["smape"],
            "val_smape": val_metrics["smape"],
            "test_smape": test_metrics["smape"],
            "train_bias": train_metrics["bias"],
            "val_bias": val_metrics["bias"],
            "test_bias": test_metrics["bias"],
        })

        print(
            f"{name:24s} | "
            f"Train MAE: {train_metrics['mae']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Test MAE: {test_metrics['mae']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_rows)

    # Merge cross-validation and holdout results
    selection_df = comparison_df.merge(cv_summary_df, on="model_name", how="left")
    selection_df["holdout_val_rank"] = selection_df["val_mae"].rank(method="dense")
    selection_df["cv_val_rank"] = selection_df["cv_val_mae_mean"].rank(method="dense")
    selection_df = selection_df.sort_values(["val_mae", "cv_val_mae_mean"]).reset_index(drop=True)

    save_dataframe(selection_df, os.path.join(ARTIFACTS_DIR, "model_selection_summary.csv"))
    save_dataframe(comparison_df, os.path.join(ARTIFACTS_DIR, "model_comparison.csv"))

    best_name = selection_df.iloc[0]["model_name"]
    best_model = fitted_models[best_name]

    print(f"\nBest model on validation: {best_name}")

    # Re-evaluate best model for final artifacts
    train_metrics, train_preds = evaluate_model(best_model, X_train, y_train, label="train")
    val_metrics, val_preds = evaluate_model(best_model, X_val, y_val, label="validation")
    test_metrics, test_preds = evaluate_model(best_model, X_test, y_test, label="test")

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Model: {best_name}")
    print(f"MAE : {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"R2  : {test_metrics['r2']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}")
    print(f"SMAPE: {test_metrics['smape']:.4f}")

    # Prediction frames
    train_pred_df = make_prediction_frame(train_df, y_train, train_preds)
    val_pred_df = make_prediction_frame(val_df, y_val, val_preds)
    test_pred_df = make_prediction_frame(test_df, y_test, test_preds)

    save_dataframe(train_pred_df, os.path.join(PREDICTIONS_DIR, "train_predictions.csv"))
    save_dataframe(val_pred_df, os.path.join(PREDICTIONS_DIR, "validation_predictions.csv"))
    save_dataframe(test_pred_df, os.path.join(PREDICTIONS_DIR, "test_predictions.csv"))

    # Error summaries
    hour_summary, zone_summary, peak_summary = summarize_errors(test_pred_df)
    cal_summary = calibration_summary(test_pred_df, n_bins=10)
    resid_summary = residual_summary(test_pred_df)

    save_dataframe(hour_summary, os.path.join(ARTIFACTS_DIR, "error_by_hour.csv"))
    save_dataframe(zone_summary, os.path.join(ARTIFACTS_DIR, "error_by_zone.csv"))
    save_dataframe(peak_summary, os.path.join(ARTIFACTS_DIR, "error_by_peak_hour.csv"))
    save_dataframe(cal_summary, os.path.join(ARTIFACTS_DIR, "calibration_summary.csv"))
    save_json(resid_summary, os.path.join(ARTIFACTS_DIR, "residual_summary.json"))

    # Save model
    joblib.dump(best_model, os.path.join(MODEL_DIR, "demand_model.pkl"))

    # Feature importance
    feature_importance_df = extract_feature_importance(best_model, ALL_FEATURES)

    if feature_importance_df is None:
        sample_size = min(5000, len(X_val))
        X_sample = X_val.sample(sample_size, random_state=42)
        y_sample = y_val.loc[X_sample.index]
        feature_importance_df = compute_permutation_importance(
            best_model,
            X_sample,
            y_sample,
            ALL_FEATURES,
            n_repeats=5
        )
        feature_importance_df.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)
        save_json(
            {"method": "permutation_importance"},
            os.path.join(ARTIFACTS_DIR, "feature_importance_meta.json")
        )
    else:
        feature_importance_df.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)
        save_json(
            {"method": "native_importance_or_coefficients"},
            os.path.join(ARTIFACTS_DIR, "feature_importance_meta.json")
        )

    # Top error cases
    top_over_predictions = test_pred_df.sort_values("error", ascending=True).head(20).copy()
    top_under_predictions = test_pred_df.sort_values("error", ascending=False).head(20).copy()

    save_dataframe(top_over_predictions, os.path.join(ARTIFACTS_DIR, "top_over_predictions.csv"))
    save_dataframe(top_under_predictions, os.path.join(ARTIFACTS_DIR, "top_under_predictions.csv"))

    # Neural network history if selected
    if best_name == "neural_network" and hasattr(best_model, "training_history_dataframe"):
        history_df = best_model.training_history_dataframe()
        save_dataframe(history_df, os.path.join(ARTIFACTS_DIR, "nn_training_history.csv"))

    # Summary metrics
    generalization_gap_mae = float(val_metrics["mae"] - train_metrics["mae"])
    generalization_gap_rmse = float(val_metrics["rmse"] - train_metrics["rmse"])
    overfitting_suspected = bool(val_metrics["mae"] > train_metrics["mae"] * 1.10)

    model_metrics = {
        "best_model": best_name,
        "comparison": comparison_rows,
        "selection_table": selection_df.to_dict(orient="records"),
        "selected_model_train_metrics": train_metrics,
        "selected_model_validation_metrics": val_metrics,
        "selected_model_test_metrics": test_metrics,
        "feature_columns": ALL_FEATURES,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "generalization_gap_mae": generalization_gap_mae,
        "generalization_gap_rmse": generalization_gap_rmse,
        "overfitting_suspected": overfitting_suspected,
        "residual_summary": resid_summary,
        "feature_importance_method": "native_or_permutation",
        "has_xgboost_branch": bool(HAS_XGB),
        "has_lightgbm_branch": bool(HAS_LGBM),
        "has_catboost_branch": bool(HAS_CATBOOST),
        "has_neural_network_branch": bool(HAS_TORCH),
        "cross_validation_profile": "walk_forward_time_series",
        "cross_validation_splits": 3,
    }

    save_json(model_metrics, os.path.join(ARTIFACTS_DIR, "model_metrics.json"))
    save_json({"feature_columns": ALL_FEATURES}, os.path.join(ARTIFACTS_DIR, "feature_columns.json"))

    # Diagnostics plots
    try:
        from visualization import plot_model_diagnostics
        plot_model_diagnostics(
            comparison_df=selection_df,
            train_pred_df=train_pred_df,
            val_pred_df=val_pred_df,
            test_pred_df=test_pred_df,
            feature_importance_df=feature_importance_df
        )
    except Exception as e:
        print(f"⚠️ Diagnostic plotting skipped: {e}")

    return best_model, test_metrics, model_metrics


def predict_demand(model, input_df: pd.DataFrame) -> np.ndarray:
    return model.predict(input_df[ALL_FEATURES].copy())


if __name__ == "__main__":
    demand_path = os.path.join(OUTPUT_DIR, "data", "demand_panel.csv")
    demand_df = pd.read_csv(demand_path)

    model, test_metrics, metrics = train_model(demand_df)

    print("\nSaved:")
    print(f"- {os.path.join(MODEL_DIR, 'demand_model.pkl')}")
    print(f"- {os.path.join(ARTIFACTS_DIR, 'model_metrics.json')}")
    print(f"- {os.path.join(PREDICTIONS_DIR, 'train_predictions.csv')}")
    print(f"- {os.path.join(PREDICTIONS_DIR, 'validation_predictions.csv')}")
    print(f"- {os.path.join(PREDICTIONS_DIR, 'test_predictions.csv')}")