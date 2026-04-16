"""
kernel_engine.py -- FP-04: Kernel Methods & Non-Linear Features

Henry Laufer'in kernel methods yaklasimi. Lineer olmayan fiyat iliskilerini yakalar.
Renaissance, modern ML devriminden cok once kernel methods kullaniyordu.

Teknikler:
  1. Kernel Regression (RBF kernel) -- non-linear return prediction
  2. SVR (Support Vector Regression)
  3. Gaussian Process regression -- uncertainty quantification
  4. Feature engineering (polynomial, interaction terms)

Refs:
  - Scholkopf & Smola (2002). Learning with Kernels. MIT Press.
  - Wilson & Adams (2013). GP Kernels for Pattern Discovery. ICML.
  - Lopez de Prado (2018), Ch. 8.
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from sklearn.svm import SVR
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, KernelPCA
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


def build_features(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Time series'ten feature matrix olustur.

    Her satir bir zaman noktasi, her sutun bir feature:
    - Son N gunluk return
    - Volatility (rolling)
    - Volume ratio
    - Momentum (1d, 5d, 10d)
    - Range features
    """
    if len(df) < lookback + 10:
        return np.array([])

    features = []
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        try:
            feat = [
                window["returns"].values[-1] if "returns" in window else 0,
                window["returns"].iloc[-5:].sum() if "returns" in window else 0,
                window["returns"].iloc[-10:].sum() if "returns" in window else 0,
                window["returns"].std() if "returns" in window else 0,
                window["volume_ratio"].values[-1] if "volume_ratio" in window else 1,
                window["overnight_return"].values[-1] if "overnight_return" in window else 0,
                window["daily_range"].values[-1] if "daily_range" in window else 0,
            ]
            features.append(feat)
        except Exception:
            continue

    return np.array(features) if features else np.array([])


def kernel_predict(df: pd.DataFrame, target: str = "returns",
                   horizon: int = 1, kernel: str = "rbf",
                   train_size: int = 100) -> dict:
    """
    Kernel regression ile gelecek return tahmin et.

    Args:
        df: OHLCV + features DataFrame
        target: Tahmin edilecek kolon
        horizon: Kac gun sonrasi (default: ertesi gun)
        kernel: "rbf", "linear", "poly"
        train_size: Training window

    Returns:
        {
            "prediction": float,
            "confidence": float,
            "signal": float in [-1, 1]
        }
    """
    if not SKLEARN_OK or len(df) < train_size + horizon:
        return {"prediction": 0, "confidence": 0, "signal": 0, "error": "insufficient_data"}

    try:
        X = build_features(df, lookback=20)
        if len(X) < train_size:
            return {"prediction": 0, "confidence": 0, "signal": 0, "error": "insufficient_features"}

        # Target: horizon gun sonraki return
        y = df[target].shift(-horizon).dropna().values

        # Align X and y
        n = min(len(X), len(y))
        X = X[:n]
        y = y[:n]

        if n < train_size:
            return {"prediction": 0, "confidence": 0, "signal": 0, "error": "insufficient_aligned"}

        # Train / predict split
        X_train = X[:-1]
        y_train = y[:-1]
        X_test = X[-1:].reshape(1, -1)

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Kernel Ridge Regression
        model = KernelRidge(kernel=kernel, alpha=0.1, gamma=0.1)
        model.fit(X_train_s, y_train)
        pred = float(model.predict(X_test_s)[0])

        # Confidence: training fit quality
        train_pred = model.predict(X_train_s)
        residuals = y_train - train_pred
        residual_std = np.std(residuals)

        # Signal: scaled prediction
        signal = float(np.clip(pred / max(residual_std, 1e-6), -1, 1))

        # Confidence: inverse residual std (normalize)
        y_std = np.std(y_train) if np.std(y_train) > 0 else 1e-6
        confidence = float(np.clip(1 - residual_std / y_std, 0, 1))

        return {
            "prediction": round(pred, 6),
            "confidence": round(confidence, 4),
            "signal": round(signal, 4),
            "kernel": kernel,
            "n_train": len(X_train),
        }
    except Exception as e:
        return {"prediction": 0, "confidence": 0, "signal": 0, "error": str(e)}


def kernel_pca_features(data: dict, n_components: int = 5) -> dict:
    """
    Multi-ticker cross-asset embedding.

    Tum ticker return serilerini alip KernelPCA ile dusuk boyutlu
    latent space'e yansit. Cross-asset iliskilerini yakalar.

    Returns:
        {ticker: [embedding_vector]}
    """
    if not SKLEARN_OK or not data:
        return {}

    try:
        # Build returns matrix
        returns_dict = {}
        for ticker, df in data.items():
            if "returns" in df.columns and len(df) > 100:
                returns_dict[ticker] = df["returns"].dropna().values[-100:]

        if len(returns_dict) < 3:
            return {}

        # Align length
        min_len = min(len(v) for v in returns_dict.values())
        matrix = np.array([returns_dict[t][-min_len:] for t in returns_dict.keys()])
        # Each row: one ticker's return time series

        # Scale
        scaler = StandardScaler()
        matrix_s = scaler.fit_transform(matrix)

        # Kernel PCA
        n_comp = min(n_components, len(returns_dict) - 1, min_len - 1)
        if n_comp < 1:
            return {}

        kpca = KernelPCA(n_components=n_comp, kernel="rbf", gamma=0.1)
        embeddings = kpca.fit_transform(matrix_s)

        return {
            ticker: [round(float(x), 4) for x in emb]
            for ticker, emb in zip(returns_dict.keys(), embeddings)
        }
    except Exception as e:
        return {"error": str(e)}


def svr_signal(df: pd.DataFrame, lookback: int = 60) -> float:
    """
    SVR ile alpha tahmini -> signal direction.

    Non-linear patterns'i yakalayan hizli bir signal.
    """
    if not SKLEARN_OK or len(df) < lookback + 10:
        return 0.0

    try:
        X = build_features(df, lookback=20)
        if len(X) < lookback:
            return 0.0

        y = df["returns"].shift(-1).dropna().values
        n = min(len(X), len(y))
        X = X[:n]
        y = y[:n]

        if n < lookback:
            return 0.0

        X_train = X[:-1]
        y_train = y[:-1]
        X_test = X[-1:].reshape(1, -1)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.001)
        model.fit(X_train_s, y_train)
        pred = float(model.predict(X_test_s)[0])

        # Scale to [-1, 1]
        std = np.std(y_train) if np.std(y_train) > 0 else 1e-6
        signal = float(np.clip(pred / std, -1, 1))
        return round(signal, 4)
    except Exception:
        return 0.0
