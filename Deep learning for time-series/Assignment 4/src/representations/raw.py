"""Raw-signal representation (Part B): the identity member of the representation family.

Vectorises a batch of series so that the raw baselines and the learned encoders of Part C
reach the probes through exactly the same path. Swapping this module for a T-Loss or TS2Vec
extractor is the only difference between Part B and Part C.

Shape convention for the whole package: input is (n_series, n_channels, length), output is
(n_series, n_features).
"""

import numpy as np


def encode(X: np.ndarray) -> np.ndarray:
    """Flatten each series to a single fixed-length vector.

    Channels are concatenated in order, so output column ``c * length + t`` holds channel
    ``c`` at time ``t``. For the univariate datasets this is the series itself; for AWR
    (9 channels) and Epilepsy (3 channels) it is the channels laid end to end.
    """
    if X.ndim != 3:
        raise ValueError(
            f"expected (n_series, n_channels, length), got shape {X.shape}"
        )
    return X.reshape(X.shape[0], -1)
