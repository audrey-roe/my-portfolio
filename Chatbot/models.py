import torch
import torch.nn as nn

# we prepare a feedforward neural net with two hidden layers that gets the bag of words as an input, one layer full 
# connnected to the number of patterns,then another layer connected to the number 
# of classes which is the output 
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU() #creating a activation function for the hidden classes in between
        
    def forward(self, x): # implementing the forward pass
        out = self.l1(x) # first linear layer
        out = self.relu(out) # activation function
        out = self.l2(out) # second linear layer
        out = self.relu(out) # second activation function
        out = self.l3(out) # thrid linear layer

        # no activation and no softmax for the third linear layer 
        # because we will still apply a cross entropy loss that will 
        # automatically apply the last activation function
        return out