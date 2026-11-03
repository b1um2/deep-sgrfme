import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy.linalg as la

class mlp(nn.Module):
    def __init__(self, num_features, num_anchors, num_classes, lr = 0.001, hidden_size = 128):
        super().__init__()

        self.num_features = num_features
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.input_height = num_features
        self.input_width = num_anchors
        self.output_height = num_anchors
        self.output_width = num_classes

        nn_in_size = self.input_height * self.input_width
        nn_out_size = num_anchors * num_classes
        # Keep the fully-connected layer small. The original formula
        # was ~43k wide, and int(nn_in_size) was still 50k wide.
        nn_hidden_size = hidden_size
        
        self.linear1 = nn.Linear(nn_in_size, nn_hidden_size)
        self.linear2 = nn.Linear(nn_hidden_size, nn_out_size)

        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, nn_input): # nn_input should be of dimensions batch * num_features * num_anchors
        nn_input = torch.flatten(nn_input)
        nn_input = self.linear1(nn_input)
        nn_input = self.linear2(nn_input)

        # Output is (num_anchors, num_classes), not (num_classes, num_anchors).
        nn_input = nn_input.reshape(self.output_height, self.output_width)
        return torch.softmax(nn_input, dim=1)

    def criterion(self, alpha, A, B, L, U, Y):
        # residual is (N, C); A and Y are (M, C).
        residual = B @ (A - Y)

        # U is diagonal. Accept either the dense (N, N) matrix or just its
        # diagonal (N,) so the model does not need to allocate N x N memory.
        if U.dim() == 1:
            weighted_residual = torch.sum(U[:, None] * residual ** 2)
        else:
            weighted_residual = torch.trace(residual.T @ U @ residual)

        return torch.trace(A.T @ L @ A) + alpha * (
            weighted_residual + torch.sum((A - Y) ** 2)
            )

    def train_iteration(self, alpha, A, B, L, U, Z):
        self.optimizer.zero_grad()

        Y = self(Z)

        loss = self.criterion(alpha, A, B, L, U, Y)

        loss.backward()
        self.optimizer.step()

        return self(Z)