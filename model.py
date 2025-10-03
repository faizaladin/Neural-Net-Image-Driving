import torch
import torch.nn as nn
import torch.nn.functional as F

class Driving(nn.Module):
    def __init__(self):
        super(Driving, self).__init__()
        # For 3x300x400 input, add padding to keep more spatial info
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Output after conv layers: (batch, 64, 38, 50)
        self.fc1 = nn.Linear(64*38*50, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc4(x))  # Output in [-1, 1]
        return x.squeeze(1)

# Example usage:
model = Driving()
img = torch.randn(1, 3, 300, 400)
out = model(img)
print(out)
