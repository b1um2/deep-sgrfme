import numpy as np


def EProjSimplex_new(v, k=1.0):
    """
    Python translation of the MATLAB EProjSimplex_new function.

    Problem:
        min 1/2 ||x - v||^2
        subject to:
            x >= 0
            sum(x) = k
    """

    v = np.asarray(v, dtype=np.float64).flatten()

    if v.size == 0:
        return np.array([]), 0

    n = len(v)

    # MATLAB:
    # v0 = v - mean(v) + k/n
    v0 = v - np.mean(v) + k / n

    # MATLAB:
    # vmin = min(v0)
    vmin = np.min(v0)

    ft = 1

    # If everything is already non-negative,
    # the projection is simply v0.
    if vmin >= 0:
        x = v0.copy()

        # Numerical correction
        s = np.sum(x)

        if s != 0:
            x = x * (k / s)

        return x, ft

    # -------------------------------------------------
    # MATLAB iterative projection
    # -------------------------------------------------

    f = 1.0
    lambda_m = 0.0

    while abs(f) > 1e-10:

        # MATLAB:
        # v1 = v0 - lambda_m
        v1 = v0 - lambda_m

        # MATLAB:
        # posidx = v1 > 0
        posidx = v1 > 0

        # Number of positive elements
        npos = np.sum(posidx)

        # -------------------------------------------------
        # IMPORTANT:
        # MATLAB would calculate:
        #
        # g = -npos
        # f = sum(v1(posidx)) - k
        # lambda_m = lambda_m - f/g
        #
        # But if npos == 0, g == 0 and Python gives
        # division by zero.
        # -------------------------------------------------

        if npos == 0:

            # No positive elements.
            #
            # Move lambda below the largest element so that
            # at least one element becomes positive.
            lambda_m = np.max(v0) - 1e-12

            # Recalculate
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)

            if npos == 0:
                # Absolute numerical fallback
                x = np.zeros(n)
                x[np.argmax(v)] = k
                return x, ft

        # MATLAB:
        # g = -npos
        g = -float(npos)

        # MATLAB:
        # f = sum(v1(posidx)) - k
        f = np.sum(v1[posidx]) - k

        # Newton update
        lambda_m = lambda_m - f / g

        ft += 1

        # MATLAB:
        # if ft > 100
        if ft > 100:
            x = np.maximum(v1, 0.0)

            s = np.sum(x)

            if s > 0:
                x = x * (k / s)
            else:
                x = np.zeros(n)
                x[np.argmax(v)] = k

            return x, ft

    # Final MATLAB:
    # x = max(v1,0)
    x = np.maximum(v1, 0.0)

    # Numerical correction to guarantee:
    # sum(x) = k
    s = np.sum(x)

    if s > 0:
        x = x * (k / s)
    else:
        x = np.zeros(n)
        x[np.argmax(v)] = k

    return x, ft