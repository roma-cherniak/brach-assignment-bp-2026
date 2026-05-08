import torch.nn as nn


CLASSES = ['A', 'B', 'C', 'D', 'nothing']

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 24, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5),
        )


    def forward(self, x):
        return self.net(x)

