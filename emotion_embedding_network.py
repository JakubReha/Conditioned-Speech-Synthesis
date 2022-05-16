from torch import nn
import torch
import math
import torch.nn.functional as F

NMELS = 80
MAX_MELSPEC_LEN = 633

def compute_padding(H, W, S=2, F=3):
    if H % S == 0:
        Ph = max(F-S, 0)
    else:
        Ph = max(F - (H % S), 0)
    if W % S == 0:
        Pw = max(F-S, 0)
    else:
        Pw = max(F - (W % S), 0)
    return nn.ZeroPad2d((int(Pw/2), math.ceil(Pw/2), int(Ph/2), math.ceil(Ph/2)))

class EmotionEmbeddingNetwork(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(EmotionEmbeddingNetwork, self).__init__()

        model = [compute_padding(NMELS, MAX_MELSPEC_LEN)]
        model += [nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(32), ]
        H = math.ceil(NMELS/2)
        W = math.ceil(MAX_MELSPEC_LEN/2)
        model += [compute_padding(H, W)]
        model += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(32), ]
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)
        model += [compute_padding(H, W)]
        model += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(64), ]
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)
        model += [compute_padding(H, W)]
        model += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(64), ]
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)
        model += [compute_padding(H, W)]
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(128), ]
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)
        model += [compute_padding(H, W)]
        model += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(128), ]
        self.model = nn.Sequential(*model)
        self.gru = nn.GRU(128*math.ceil(NMELS/64), 128, batch_first=True)
        self.fc0 = nn.Linear(128, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = x.unsqueeze(1)
        batch = x.shape[0]
        x = self.model(x)
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
        x = self.gru(x)
        x = torch.reshape(x[1].squeeze(), (batch, -1))
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        embedding = self.fc2(x)
        x = F.relu(embedding)
        x = self.fc3(x)
        return x, embedding

if __name__ == '__main__':
    dim = 91
    max_l = 302
    batch = 256
    a = torch.rand((batch, 1, dim, max_l))
    model = EmotionEmbeddingNetwork()
    out = model(a)
    print(out.shape)