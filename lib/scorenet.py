import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(2, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.linear(x)
    
class ModelAnneal(nn.Module):
    def __init__(self, num_classes=10, num_features=1):
        super(ModelAnneal, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(3, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 2),
        )

        self.embed = nn.Embedding(num_classes, num_features)

    def forward(self, x, label):
        emb = self.embed(label)
        return self.linear(torch.cat([x, emb], dim=1))