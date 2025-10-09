import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, check_is_fitted
from sklearn.utils._param_validation import StrOptions


class FourierTransformer(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "n_harmonics": [int],
        "smooth": ["boolean"],
        "detrend": ["boolean"],
        "window": [int],
    }

    def __init__(
        self,
        n_harmonics: int = 10,
        smooth: bool = False,
        detrend: bool = False,
        window: int = 6,
    ):
        self.n_harmonics = n_harmonics
        self.smooth = smooth
        self.detrend = detrend
        self.window = window

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse=False, cast_to_ndarray=False)  # type: ignore
        self.length_ = len(X)

        self.amps_ = {}
        self.phases_ = {}
        self.freqs_ = {}
        self.columns_ = X.columns.tolist()

        for col in self.columns_:
            signal = X[col]

            if self.smooth:
                trend = signal.rolling(window=self.window, min_periods=1).mean()
                signal = signal - trend if self.detrend else trend

            # FFT
            signal_fft = fft(signal.to_numpy())
            frequencies = fftfreq(self.length_)

            amplitudes = 2 * np.abs(signal_fft) / self.length_
            phases = np.angle(signal_fft)

            indices = np.argsort(amplitudes)[::-1][: self.n_harmonics]

            self.amps_[col] = amplitudes[indices][:, None]  # shape (K, 1)
            self.phases_[col] = phases[indices][:, None]  # shape (K, 1)
            self.freqs_[col] = frequencies[indices][:, None]  # shape (K, 1)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, cast_to_ndarray=False)  # type: ignore

        transformed = {}
        t = np.arange(len(X))
        t_grid = t[None, :]  # shape: (1, T)

        for col in X.columns:
            if col not in self.amps_:
                raise ValueError(f"No fitted transformer found for column '{col}'")

            omega = 2 * np.pi * self.freqs_[col] * t_grid
            extrapolated = np.sum(self.amps_[col] * np.cos(omega + self.phases_[col]), axis=0)
            transformed.update(
                {f"fourier_{col}": np.clip(extrapolated, 0, extrapolated.max())}
            )

        return pd.DataFrame(transformed, index=X.index)
