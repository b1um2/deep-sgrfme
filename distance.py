import numpy as np

def L2_distance_1(a, b):

    # MATLAB uses double precision by default
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)

    ab = a.T @ b

    d = (
        aa[:, np.newaxis]
        + bb[np.newaxis, :]
        - 2.0 * ab
    )

    d = np.maximum(d, 0.0)

    return d