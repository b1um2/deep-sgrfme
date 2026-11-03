import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy.linalg as la

class mlp(nn.Module):
    def __init__(self, num_features, num_anchors, num_classes, lr = 0.001, hidden_size = 256):
        super().__init__()

        self.num_features = num_features
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.input_height = num_features
        self.input_width = num_anchors
        self.output_height = num_anchors
        self.output_width = num_classes

        # Each anchor is mapped independently from its features to its classes.
        self.linear1 = nn.Linear(num_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, nn_input): # nn_input should be of dimensions batch * num_features * num_anchors
        if nn_input.dim() == 3:
            x = nn_input.squeeze(0).T            # (anchors, features)
        elif nn_input.dim() == 2:
            x = nn_input.T                       # (anchors, features)
        else:
            x = nn_input

        x = torch.relu(self.linear1(x))
        x = self.linear2(x)                      # (anchors, classes)
        return torch.softmax(x, dim=1)

    def criterion(self, alpha, A, B, L, U, Y, Y_anchor):
        # A is (num_anchors, num_classes), B is (num_samples, num_anchors),
        # and Y is the (num_samples, num_classes) ground-truth label matrix.
        residual = B @ A - Y

        # U is diagonal. Accept either the dense (N, N) matrix or just its
        # diagonal (N,) so the model does not need to allocate N x N memory.
        if U.dim() == 1:
            weighted_residual = torch.sum(U[:, None] * residual ** 2)
        else:
            weighted_residual = torch.trace(residual.T @ U @ residual)

        # Y_anchor is the other network's (num_anchors, num_classes) output.
        return torch.trace(A.T @ L @ A) + alpha * (
            weighted_residual + torch.sum((A - Y_anchor) ** 2)
            )

    def train_iteration(self, alpha, A_other, B, L, U, Z, Y):
        self.optimizer.zero_grad()

        Y_anchor = self(Z)

        # Match the original SGRFME: the labeled fit and graph term use A
        # (the GCN output); the MLP output only enters the consistency term.
        loss = self.criterion(alpha, A_other, B, L, U, Y, Y_anchor)

        loss.backward()
        self.optimizer.step()

        return self(Z)