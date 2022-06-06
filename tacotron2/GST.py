from torch import nn
import torch
import math
import torch.nn.init as init

NMELS = 80

class GST(nn.Module):
    def __init__(self, n_heads=4, n_tokens=10, norm_layer=nn.BatchNorm2d):
        super(GST, self).__init__()
        self.n_heads = n_heads
        self.n_tokens = n_tokens
        self.tokens = nn.Parameter(torch.FloatTensor(n_tokens, 256), requires_grad=False)
        init.normal_(self.tokens, mean=0, std=1)
        self.tanh = nn.Tanh()
        self.attention = nn.MultiheadAttention(128, 4, vdim=256, kdim=256, batch_first=True)
        model = [nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(32), ]
        model += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(32), ]
        model += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(64), ]
        model += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(64), ]
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(128), ]
        model += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True), ]
        model += [nn.ReLU(True), ]
        model += [norm_layer(128), ]
        self.model = nn.Sequential(*model)
        self.gru = nn.GRU(128*math.ceil(NMELS/64), 128, batch_first=True)


    def forward(self, x):
        x = x.unsqueeze(1)
        batch = x.shape[0]
        x = self.model(x)
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
        x = self.gru(x)
        x = torch.reshape(x[1].squeeze(), (batch, -1))
        tokens = self.tanh(self.tokens)
        _, weights = self.attention(x, tokens, tokens)
        x = weights @ tokens
        return x

    def token_inference(self, weights):
        weights = weights.unsqueeze(0)
        tokens = self.tanh(self.tokens)
        x = weights @ tokens
        return x

if __name__ == '__main__':
    dim = 80
    max_l = 302
    batch = 8
    a = torch.rand((batch, dim, max_l))
    model = GST()
    out = model(a)
    print(out.shape)