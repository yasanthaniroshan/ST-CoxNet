import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,number_of_features: int,number_of_classes: int):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(number_of_features, 128),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc4 = nn.Linear(32, number_of_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x.squeeze()