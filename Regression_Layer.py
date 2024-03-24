# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:12:02 2024

@author: user
"""

# import torch
# import torch.nn as nn
# import torch.optim as optim

# class RegressionNN(nn.Module):
#     def __init__(self, input_size_psi, input_size_k):
#         super(RegressionNN, self).__init__()
#         # Define the layers of the neural network
#         self.fc_psi = nn.Linear(input_size_psi, 64)  # Fully connected layer for input size 1201 using 64 neurons
#         self.fc_k = nn.Linear(input_size_k, 64)  # Fully connected layer for input size 1
#         self.fc_combined = nn.Linear(64 * 2, 1)   # Fully connected layer to combine inputs and produce output

#     def forward(self, x_n, x_m):
#         # Forward pass through the neural network
#         out_n = torch.relu(self.fc_n(x_n))  # Pass input psi through fully connected layer and apply ReLU activation
#         out_m = torch.relu(self.fc_m(x_m))  # Pass input kappa  through fully connected layer and apply ReLU activation
#         combined_out = torch.cat((out_n, out_m), dim=1)  # Concatenate the outputs from the two branches
#         output = self.fc_combined(combined_out)  # Pass the concatenated output through the final fully connected layer
#         return output

# input_size_n = 1202  # Size of the first input list
# input_size_m = 1     # Size of the second input list
# model = RegressionNN(input_size_n, input_size_m)  # Create an instance of the regression neural network
# print(model)  # Print the architecture of the neural network


import torch
import torch.nn as nn
import torch.optim as optim


class AveragingNetwork(nn.Module):
    def __init__(self, input_size_1, input_size_N, hidden_size, output_size):
        super(AveragingNetwork, self).__init__()
        self.input_network_1 = nn.Linear(input_size_1, hidden_size)
        self.input_network_N = nn.Linear(input_size_N, hidden_size)
        self.output_network = nn.Linear(hidden_size, output_size)

    def forward(self, input_1, input_N):
        # Pass input 1 through input network and apply ReLU activation
        representation_1 = torch.relu(self.input_network_1(input_1))
        
        # Pass input N through input network and apply ReLU activation
        representation_N = torch.relu(self.input_network_N(input_N))

        # Average the representations across all inputs
        avg_representation = (representation_1 + representation_N) / 2.0

        # Pass the averaged representation through the output network
        output = self.output_network(avg_representation)

        return output

# Example usage:
input_size_1 = 1  # Size of input 1
input_size_N = 10  # Size of input N
hidden_size = 64  # Size of the hidden layer
output_size = 1  # Size of the output

# Example usage:

# # Define your dataset and DataLoader
# train_dataset = YourDataset(...)  # Define your dataset
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     model.train()  # Set model to training mode
#     for inputs_n, inputs_m, targets in train_loader:
#         # Zero the gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs_n, inputs_m)

#         # Compute loss
#         loss = criterion(outputs, targets)

#         # Backward pass
#         loss.backward()

#         # Update weights
#         optimizer.step()

#     # Optionally, evaluate model on validation set after each epoch
