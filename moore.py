import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def moore_mask(degree: int) -> torch.Tensor:
    size = 1 + degree + (degree - 1) * degree
    mask = torch.ones(size, size)
    mask[-1, -1] = 0
    mask[:degree * (degree - 1), -1] = 0
    mask[:-1, (degree - 1) * degree:-1] = 0
    for i in range(degree):
        mask[i * (degree - 1): (i + 1) * (degree - 1), (degree - 1) * degree + i] = 1
        mask[i * (degree - 1): (i + 1) * (degree - 1), i * (degree - 1): (i + 1) * (degree - 1)] = 0
    mask = mask.triu()
    mask = mask + mask.T
    return mask

def moore_target(degree: int) -> torch.Tensor:
    size = 1 + degree + (degree - 1) * degree
    return torch.eye(size) * (degree - 1) + torch.ones(size, size)


degree = 7
batch_size = 16

torch.set_printoptions(precision=1)

partial_optimizer = partial(torch.optim.Adam, lr=0.1)

def moore_adj_mat(params: torch.Tensor, degree: int, mask: torch.Tensor) -> torch.Tensor:
    adj_mat = F.sigmoid(params)
    adj_mat = F.pad(adj_mat, (0, 1 + degree, 0, 1 + degree), value=1.0).triu()
    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat * mask
 
mask = moore_mask(degree).detach().cuda()
target = moore_target(degree).detach().cuda()

class MooreModel(nn.Module):
    def __init__(self, degree: int):
        super(MooreModel, self).__init__()
        self.degree = degree
        self.mask = moore_mask(degree).detach().cuda()

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        adj_mat = F.sigmoid(params)
        adj_mat = F.pad(adj_mat, (0, 1 + self.degree, 0, 1 + self.degree), value=1.0).triu()
        adj_mat = adj_mat + adj_mat.transpose(-2, -1)
        adj_mat = adj_mat * self.mask
        return adj_mat

model = MooreModel(degree).cuda()

params = nn.Parameter(torch.randn(batch_size, degree * (degree - 1), degree * (degree - 1)).cuda())
optimizer = partial_optimizer(params=[params])

for step in range(10000):
    optimizer.zero_grad()
    adj_mat = model(params)
    hat = torch.matmul(adj_mat, adj_mat) + adj_mat
    loss_batch = F.mse_loss(hat, target, reduction='none').mean(dim=(1, 2))
    loss = loss_batch.mean()
    loss.backward()
    optimizer.step()
    print(f'Step {step}: Loss = {loss_batch.min().item():.4f}')

adj_mat = torch.round(model(params).detach())
hat = torch.matmul(adj_mat, adj_mat) + adj_mat
print(adj_mat)
print((hat - target).sum(dim=(1, 2)).min().item())

