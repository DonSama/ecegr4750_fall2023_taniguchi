import torch
import torch.nn as nn

# adapted from LearningModel from prior lab
class FullyConnectedNetwork(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int):
        super(FullyConnectedNetwork, self).__init__()
        assert input_dim > 0, "Input dimension must be a positive integer"
        assert hidden_dim > 1, "Hidden dimensions must be an integer greater than 1"
        self.linear1 = nn.Linear(in_features = input_dim, out_features = hidden_dim)
        self.linear2 = nn.Linear(in_features = hidden_dim, out_features = round(hidden_dim//2))
        self.linear3 = nn.Linear(in_features = round(hidden_dim//2), out_features = 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)       
        return x



# have students write this one themselves
class FCNClassifier(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(FCNClassifier, self).__init__()
        assert input_dim > 0, "Input dimension must be a positive integer"
        assert hidden_dim > 1, "Hidden dimensions must be an integer greater than 1"
        assert output_dim > 0, "Output dimension must be a positive integer"
        self.linear1 = nn.Linear(in_features = input_dim, out_features = hidden_dim)
        self.linear2 = nn.Linear(in_features = hidden_dim, out_features = round(hidden_dim//2))
        self.linear3 = nn.Linear(in_features = round(hidden_dim//2), out_features = output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)     
        x = torch.sigmoid(x)  
        return x

# image CNN
# made with help of ChatGPT, but failed to be implemented
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256 * 8 * 8, 256)
        self.dense2 = nn.Linear(256 * 8 * 8, 256)
        
        # Output layer for regression
        self.dropout = nn.Dropout(0.3)
        self.age_out = nn.Linear(256, 1)

    def forward(self, x):
        # Convolutional layers
        x = self.maxpool1(nn.functional.relu(self.conv1(x)))
        x = self.maxpool2(nn.functional.relu(self.conv2(x)))
        x = self.maxpool3(nn.functional.relu(self.conv3(x)))
        x = self.maxpool4(nn.functional.relu(self.conv4(x)))

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        dense1 = nn.functional.relu(self.dense1(x))
        dense2 = nn.functional.relu(self.dense2(x))

        # Dropout layer
        dropout = self.dropout(dense1)

        # Output layer for regression
        age_out = self.age_out(dropout)

        return age_out