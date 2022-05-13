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

class ReferenceEncoder(nn.Module):
    def __init__(self, dim=NMELS, max_l=MAX_MELSPEC_LEN, norm_layer=nn.BatchNorm2d):
        super(ReferenceEncoder, self).__init__()
        net = [compute_padding(dim, max_l)]
        net += [nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0, bias=True), ]
        net += [nn.ReLU(True), ]
        net += [norm_layer(32), ]
        H = math.ceil(dim/2)
        W = math.ceil(max_l/2)
        net += [compute_padding(H, W)]
        net += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, bias=True), ]
        net += [nn.ReLU(True), ]
        net += [norm_layer(32), ]
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)
        net += [compute_padding(H, W)]
        net += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0, bias=True), ]
        net += [nn.ReLU(True), ]
        net += [norm_layer(64), ]
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)
        net += [compute_padding(H, W)]
        net += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, bias=True), ]
        net += [nn.ReLU(True), ]
        net += [norm_layer(64), ]
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)
        net += [compute_padding(H, W)]
        net += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=True), ]
        net += [nn.ReLU(True), ]
        net += [norm_layer(128), ]
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)
        net += [compute_padding(H, W)]
        net += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=True), ]
        net += [nn.ReLU(True), ]
        net += [norm_layer(128), ]
        self.net = net
    
    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.net:
            out = layer(x)
            x = out
        return out

class Classifier(nn.Module):
    def __init__(self, dim=NMELS):
        super(Classifier, self).__init__()
        self.gru = nn.GRU(128*math.ceil(dim/64), 128, batch_first=True)
        self.fc0 = nn.Linear(128, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 7)
    
    def forward(self, x, batch_size):
        x = self.gru(x)
        x = torch.reshape(x[1].squeeze(), (batch_size, -1))
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        embedding = self.fc2(x)
        x = F.relu(embedding)
        x = self.fc3(x)
        return x, embedding


class EmotionEmbeddingNetwork(nn.Module):
    def __init__(self, dim=NMELS, max_l=MAX_MELSPEC_LEN, norm_layer=nn.BatchNorm2d):
        super(EmotionEmbeddingNetwork, self).__init__()
        self.reference_encoder = ReferenceEncoder(dim=dim, max_l=max_l, norm_layer=norm_layer)
        self.classifier = Classifier(dim=dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.reference_encoder(x)
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
        x, embedding = self.classifier(x, batch_size)
        return x, embedding

if __name__ == '__main__':
    dim = 91
    max_l = 302
    batch = 256
    a = torch.rand((batch, 1, dim, max_l))
    model = EmotionEmbeddingNetwork(dim, max_l)
    out = model(a)
    print(out.shape)