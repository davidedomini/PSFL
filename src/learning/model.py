from torch import nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class MLP(nn.Module):

    def __init__(self, h1=128):
        super().__init__()
        self.fc1 = nn.Linear(28*28, h1)
        self.fc2 = nn.Linear(h1, 27)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def init_mobilenet():
    model = mobilenet_v2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 100)
    )
    return model