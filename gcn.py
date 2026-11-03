import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class gcn(nn.Module):
    def __init__(self, num_features, num_anchors, num_classes, lr=0.001):
        super().__init__()

        self.num_features = num_features
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.input_height = num_features
        self.input_width = num_anchors
        self.output_height = num_anchors
        self.output_width = num_classes

        total_input_elements = self.input_height * self.input_width
        conv_hidden_size = 2
        conv_out_size = 4

        nn_in_size = total_input_elements * conv_out_size
        nn_out_size = num_anchors * num_classes
        #nn_hidden_size = int(nn_in_size*2/3 + nn_out_size)
        nn_hidden_size = int(nn_in_size / conv_out_size)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv_hidden_size,
            kernel_size=3,
            padding=1  # keep dimensions (height, width) the same
        )

        self.conv2 = nn.Conv2d(
            in_channels=conv_hidden_size,
            out_channels=conv_out_size,
            kernel_size=3,
            padding=1  # keep dimensions (height, width) the same
        )
        
        self.linear1 = nn.Linear(nn_in_size, nn_hidden_size)
        self.linear2 = nn.Linear(nn_hidden_size, nn_out_size)

        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, nn_input): # nn_input should be Z @ S for GCN message passing
        nn_input = self.conv1(nn_input)
        nn_input = self.conv2(nn_input)
        nn_input = torch.flatten(nn_input, start_dim=1)
        nn_input = self.linear1(nn_input)
        nn_input = self.linear2(nn_input)

        return torch.softmax(nn_input.reshape(self.output_width, self.output_height), dim=0)

    def criterion(self, alpha, A, B, L, U, Y):
        return torch.trace(A.T @ L @ A) + alpha * (
            torch.trace((B @ A - Y).T @ U @ (B @ A - Y)) + torch.sum((A - Y) ** 2)
            )
    
    def train_iteration(self, alpha, Y, B, L, U, Z, S):
        self.optimizer.zero_grad()

        A = self(Z @ S)

        loss = self.criterion(alpha, A, B, L, U, Y)

        loss.backward()
        self.optimizer.step()

        return self(Z)