import torch
class StyleLoss:
    def __call__(self, R, S):
        R = R.reshape(R.shape[0], 16, 16)
        S = S.reshape(S.shape[0], 16, 16)
        G = torch.matmul(R.transpose(1, 2), R)
        I = torch.matmul(S.transpose(1, 2), S)
        loss = torch.sum((I - G)**2)/(2*16*16)**2
        loss = loss / R.shape[0]
        return loss


if __name__ == '__main__':
    dim = 256
    batch = 6
    a = torch.rand((batch, dim))
    b = torch.rand((batch, dim))
    loss = StyleLoss()
    J = loss(a, b)