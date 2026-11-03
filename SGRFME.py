import distance
import numpy as np
from MLAN import MLAN_Solution
import distance
import numpy as np
import torch

def SGRFME_model(
    anchors,
    xtrain,
    y_train,
    k,
    X_labeled,
    X_test,
    Y_test,
    gamma,
    lambdah,
    rho,
    gcn,
    mlp,
    epochs = 40
):

    z = anchors.T
    #print("z shape:", z.shape)

    num_anchors = anchors.shape[0]
    num_labeled = X_labeled.shape[0]
    #Estimate B matrix 
    #B = Data-to-anchor graph matrix ∈ N×m

    #initialize weighted_distX
    distX = distance.L2_distance_1(z, xtrain.T).T
    idx = np.argsort(distX, axis=1)
    distXs = np.take_along_axis(distX, idx, axis=1)


    # Number of samples and anchors
    num_samples = xtrain.shape[0]
    num_anchors = anchors.shape[0]

    

    B = np.zeros((num_samples, num_anchors))

    eps = 1e-10
    
    for i in range(num_samples):

        ids = idx[i, 1:k+2]

        di = distXs[i, 1:k+2]
 

        B[i, ids] = (
            (di[-1] - di) /
            (k * di[-1] - np.sum(di[:-1]) + eps)
        )
         
     
    Wtilde = B.T @ B


    # Ha matrix
    Ha = np.eye(num_anchors) - (1 / num_anchors) * np.ones(
    (num_anchors, num_anchors)
    )


    # Anchor distance matrix
    dist_anchors = distance.L2_distance_1(
        z,
        z
    ).T

    YB = np.argmax(y_train.T, axis=1)

    # Number of samples
    num_samples = xtrain.shape[0]

    # Number of labeled samples
    num_labeled = X_labeled.shape[0]


    # ============================
    # Construct U diagonal
    # ============================

    u_values = np.zeros(num_samples)

    # First num_labeled samples are labeled
    u_values[:num_labeled] = 1e9

    # ============================
    # Compute BtUY
    # ============================

    # Anchors
    Z = anchors.T      # (50,1000)

    Wtilde = B.T @ B

    Wtildes = []

    Wtildes.append(Wtilde.copy())

    acc_train = np.zeros(10)
    acc_test = np.zeros(10)

    # convert constant numpy arrays to pytorch once
    Z_tensor = torch.tensor(Z, dtype=torch.float32)
    B_tensor = torch.tensor(B, dtype=torch.float32)
    # U is diagonal, so pass only its diagonal instead of a dense N x N matrix.
    U_tensor = torch.tensor(u_values, dtype=torch.float32)

    for iter in range(10):

        # ============================
        # Graph fixed, estimate A, W, b
        # ============================

        # matrix L (recomputed from the current Wtilde each iteration)
        degree = np.sum(Wtilde, axis=1)

        # avoid zero degree
        degree[degree==0] = np.finfo(float).eps

        D = np.diag(degree)

        L = D - Wtilde

        # these change when Wtilde changes, so build them per iteration
        L_tensor = torch.tensor(L, dtype=torch.float32)
        ZS_tensor = torch.tensor(Z @ Wtilde, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (batch, channel, features, anchors)

        try:
            for epoch in range(epochs):
                Y = mlp(Z_tensor) # initial guess for Y

                A = gcn.train_iteration(1, Y, B_tensor, L_tensor, U_tensor, Z_tensor, ZS_tensor)
                Y = mlp.train_iteration(1, A, B_tensor, L_tensor, U_tensor, Z_tensor)

        except Exception as e:
            print("Error calculating A or Y with exception: " + str(e))
            return None, None

        # gcn returns a torch tensor; the rest of the algorithm uses numpy.
        A = A.detach().cpu().numpy()

        M2 = (
            z @ Ha @ z.T
            + gamma * np.eye(z.shape[0])
        )

        try:
            W = np.linalg.solve(
                M2,
                z @ Ha @ A
            )

        except np.linalg.LinAlgError:
            print("Singular M2 - skip this parameter combination")
            return None, None

        ones_anchor = np.ones((num_anchors, 1))

        b = (
            A.T @ ones_anchor
            - W.T @ anchors.T @ ones_anchor
        ) / num_anchors


        F = B @ A
        FB = np.argmax(F, axis=1)

        lbldiff = FB - YB

        acc_train[iter] = np.mean(
            lbldiff[num_labeled:] == 0
        )

         

        # ==========================================
        # Testing prediction
        # ==========================================

        FTEST = (W.T @ X_test.T) + b


        # MATLAB max(FTEST) along columns
        FT = np.argmax(FTEST, axis=0)


        # Remove temporary variable (optional)
        del FTEST


        # Compare predicted labels with true labels
        lbltdiff = FT - Y_test


        # Test accuracy
        acc_test[iter] = np.mean(
            lbltdiff == 0
        )

         

        # ==========================================
        # A, W, b fixed, estimate Wtilde
        # ==========================================

        # Distance between A rows
        dist_A = distance.L2_distance_1(
            A.T,
            A.T
        ).T

        # Combine distances
        G = dist_A + lambdah * dist_anchors

        # Initialize Wtilde
        Wtilde = np.zeros(
            (num_anchors, num_anchors)
        )


        Wtilde = MLAN_Solution(
            G / (2 * rho),
            Wtilde,
            k
        )
        
        # Symmetric graph
        Wtilde = (Wtilde + Wtilde.T) / 2

         

        Wtildes.append(Wtilde.copy())

        diff = np.sum(np.abs(Wtildes[-2] - Wtildes[-1]))
        norm = np.sum(Wtildes[-2])

        if norm > 0 and diff / norm < 0.01:
            break

    final_train_acc = acc_train[iter]
    final_test_acc = acc_test[iter]

    return final_train_acc, final_test_acc

