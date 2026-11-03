import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class gcn(nn.Module):
    def __init__(self, num_features, num_anchors, num_classes, lr=0.001, hidden_size=256, conv_channels=32):
        super().__init__()

        self.num_features = num_features
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.input_height = num_features
        self.input_width = num_anchors
        self.output_height = num_anchors
        self.output_width = num_classes

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1  # keep dimensions (height, width) the same
        )

        self.conv2 = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1  # keep dimensions (height, width) the same
        )

        # Each anchor is predicted independently from its
        # (conv_channels x num_features) local representation.
        self.linear1 = nn.Linear(conv_channels * num_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, nn_input): # nn_input should be (X @ B) @ S for GCN message passing
        # Conv2d needs (batch, channels, height, width). Accept the 2D
        # (features, anchors) input as well as an already-batched tensor.
        if nn_input.dim() == 2:
            nn_input = nn_input.unsqueeze(0).unsqueeze(0)
        elif nn_input.dim() == 3:
            nn_input = nn_input.unsqueeze(1)

        x = torch.relu(self.conv1(nn_input))
        x = torch.relu(self.conv2(x))

        x = x.squeeze(0)                         # (conv_channels, features, anchors)
        x = x.permute(2, 0, 1)                   # (anchors, conv_channels, features)
        x = x.reshape(x.shape[0], -1)            # (anchors, conv_channels * features)

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
    
    def train_iteration(self, alpha, Y_anchor, B, L, U, ZS, Y):
        self.optimizer.zero_grad()

        A = self(ZS)

        loss = self.criterion(alpha, A, B, L, U, Y, Y_anchor)

        loss.backward()
        self.optimizer.step()

        return self(ZS)