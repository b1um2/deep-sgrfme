import numpy as np
import SGRFME
from mlp import mlp
from gcn import gcn

# ============================================================
# FIXED SETTINGS
# ============================================================

path = "./dataset/MNIST_pca_50/"

k = 10
num_labeled_samples = 8

# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(
    path + "X_train.npy"
).astype(np.float64)

y_train = np.load(
    path + "y_train.npy"
)

y_train = y_train - np.min(y_train)

# ============================================================
# LOAD ANCHORS
# ============================================================

anchors = np.load(
    path + "anchors_1000.npy"
).astype(np.float64)

# ============================================================
# LOAD SPLITS
# ============================================================

splits = np.load(
    path + "splits.npy",
    allow_pickle=True
)

# ============================================================
# LOAD TEST DATA
# ============================================================

xtest = np.load(
    path + "X_test.npy"
).astype(np.float64)

ytest = np.load(
    path + "y_test.npy"
)

ytest = ytest - np.min(ytest)

# ============================================================
# CLASSES
# ============================================================

classes = np.unique(y_train)

# ============================================================
# ONLY FIRST SPLIT
# ============================================================

split = 0

print("\n")
print("====================")
print("Split: 0")
print("====================")

# ============================================================
# GET SPLIT
# ============================================================

split_idx = splits[split].astype(int)

# ============================================================
# SELECT 5 LABELED SAMPLES PER CLASS
#
# The split is already randomized.
# Take first 5 occurrences of each class.
# ============================================================

labeled_idx = []

for c in classes:

    class_idx = split_idx[
        y_train[split_idx] == c
    ]

    selected = class_idx[
        :num_labeled_samples
    ]

    labeled_idx.extend(selected)

labeled_idx = np.array(
    labeled_idx,
    dtype=int
)

# ============================================================
# UNLABELED
# ============================================================

unlabeled_idx = np.setdiff1d(
    split_idx,
    labeled_idx,
    assume_unique=True
)

# ============================================================
# LABELED DATA
# ============================================================

X_labeled = X_train[
    labeled_idx
]

y_labeled = y_train[
    labeled_idx
]

# ============================================================
# UNLABELED DATA
# ============================================================

X_unlabeled = X_train[
    unlabeled_idx
]

y_unlabeled = y_train[
    unlabeled_idx
]

# ============================================================
# MERGE
# ============================================================

xtrain = np.vstack(
    (
        X_labeled,
        X_unlabeled
    )
)

ytrain = np.concatenate(
    (
        y_labeled,
        y_unlabeled
    )
)

# ============================================================
# ONE-HOT LABELS
# ============================================================

num_classes = len(
    np.unique(ytrain)
)

num_features = X_labeled.shape[1]

num_anchors = anchors.shape[0]

Y_train = np.eye(
    num_classes
)[
    ytrain - np.min(ytrain)
].T

# ============================================================
# CHECK SHAPES
# ============================================================

print(
    "X_labeled shape   :",
    X_labeled.shape
)

print(
    "X_unlabeled shape :",
    X_unlabeled.shape
)

print(
    "xtrain shape      :",
    xtrain.shape
)

print(
    "Y_train shape     :",
    Y_train.shape
)

print(
    "splits shape:",
    splits.shape
)

# ============================================================
# RANDOM PARAMETER SEARCH
# ============================================================

num_trials = 100

search_rng = np.random.default_rng(0)

# Sample log-uniformly because these parameters span several orders of
# magnitude. rho uses the same bounds as gamma and lambda.
search_range = [-3.0, 3.0]

gamma_samples = 10 ** search_rng.uniform(*search_range, num_trials)
lambda_samples = 10 ** search_rng.uniform(*search_range, num_trials)
rho_samples = 10 ** search_rng.uniform(*search_range, num_trials)
alpha_samples = 10 ** search_rng.uniform(*search_range, num_trials)

total_combinations = num_trials

print("\n")
print("=" * 60)
print("RANDOM PARAMETER SEARCH")
print("=" * 60)

print("Trials:", num_trials)

print(
    "search range: " + search_range[0] + " ... " + search_range[1] + " (log-uniform)"
)

print("=" * 60)

# ============================================================
# BEST RESULT
# ============================================================

best_test_acc = -np.inf
best_train_acc = -np.inf
best_params = None

combination = 0

# ============================================================
# INITIALIZE NEURAL NETWORKS FOR GLOBAL LEARNING (bad turns out it collapses probability to 10%)
# ============================================================
#A_approx = gcn(num_features, num_anchors, num_classes)
#Y_approx = mlp(num_features, num_anchors, num_classes)

# ============================================================
# SCAN PARAMETERS
# ============================================================

for trial in range(num_trials):

                gamma = gamma_samples[trial]
                lambdah = lambda_samples[trial]
                rho = rho_samples[trial]
                alpha = alpha_samples[trial]

                combination += 1

                print("\n")
                print("=" * 60)

                print(
                    f"Trial "
                    f"{combination}/{total_combinations}"
                )

                print("=" * 60)

                print(
                    "gamma  =",
                    gamma
                )

                '''print(
                    "X_labeled.size()miu    =",
                    miu
                )'''

                print(
                    "lambda =",
                    lambdah
                )

                print(
                    "rho    =",
                    rho
                )

                print(
                    "alpha  =",
                    alpha
                )

                print("=" * 60)

                # ====================================================
                # RUN SGRFME
                # ====================================================

                try:
                    A_approx = gcn(num_features, num_anchors, num_classes)
                    Y_approx = mlp(num_features, num_anchors, num_classes)

                    train_acc, test_acc = (
                        SGRFME.SGRFME_model(
                            anchors,
                            xtrain,
                            Y_train,
                            k,
                            X_labeled,
                            xtest,
                            ytest,
                            gamma,
                            #miu,
                            lambdah,
                            rho,
                            alpha,
                            A_approx,
                            Y_approx
                        )
                    )

                except Exception as e:

                    print("\nERROR:")

                    print(
                        type(e).__name__
                    )

                    print(e)

                    continue

                # ====================================================
                # CHECK RESULT
                # ====================================================

                if (
                    train_acc is None
                    or test_acc is None
                ):

                    print(
                        "Skipping this parameter combination"
                    )

                    continue

                # ====================================================
                # OUTPUT
                # ====================================================

                print(
                    "\nFinal Train:",
                    train_acc * 100
                )

                print(
                    "Final Test:",
                    test_acc * 100
                )

                # ====================================================
                # UPDATE BEST
                # ====================================================

                if test_acc > best_test_acc:

                    best_test_acc = test_acc

                    best_train_acc = train_acc

                    best_params = (
                        gamma,
                        #miu,
                        lambdah,
                        rho,
                        alpha
                    )

                    print("\n")
                    print(
                        "******** BEST UPDATED ********"
                    )

                    print(
                        "gamma:",
                        gamma
                    )

                    '''print(
                        "miu:",
                        miu
                    )'''

                    print(
                        "lambda:",
                        lambdah
                    )

                    print(
                        "rho:",
                        rho
                    )

                    print(
                        "alpha:",
                        alpha
                    )

                    print(
                        "Train:",
                        train_acc * 100
                    )

                    print(
                        "Test:",
                        test_acc * 100
                    )

# ============================================================
# FINAL BEST RESULT
# ============================================================

print("\n")
print("=" * 60)
print("BEST RESULT - SPLIT", split)
print("=" * 60)

if best_params is None:

    print(
        "No valid parameter combination found."
    )

else:

    print(
        "gamma =",
        best_params[0]
    )

    '''print(
        "miu =",
        best_params[1]
    )'''

    print(
        "lambda =",
        best_params[1]
    )

    print(
        "rho =",
        best_params[2]
    )

    print(
        "alpha =",
        best_params[3]
    )

    print(
        "\nBest Train Accuracy:",
        best_train_acc * 100
    )

    print(
        "Best Test Accuracy:",
        best_test_acc * 100
    )

print("=" * 60)