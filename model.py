import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12544, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)  # Apply dropout after pooling
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)  # Apply dropout after pooling
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout(x)  # Apply dropout after pooling
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout before final layers
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout before final layers
        x = self.sigmoid(self.fc3(x))
        return x