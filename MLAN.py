import numpy as np
from EProjSimplex_new import EProjSimplex_new


def MLAN_Solution(G, S, k):

    idx = np.argsort(G, axis=1)

    num_samples, num_anchors = S.shape

    for i in range(num_samples):

        idx0 = idx[i, 1:k+2]

        ad = -G[i, idx0]

        x, _ = EProjSimplex_new(ad)

        S[i, idx0] = x

    return S