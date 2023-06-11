from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.Conv(x)
        x = self.fc(x)
        return x