import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm

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


#degree = 57
#batch_size = 1
#num_steps = 50000
degree = 7
batch_size = 8
num_steps = 10000


torch.set_printoptions(precision=1, edgeitems=1000, linewidth=1000)

partial_optimizer = partial(torch.optim.Adam, lr=4e-1)

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
target = target[None,:,:].expand(batch_size, -1, -1)

params = nn.Parameter(torch.randn(batch_size, degree * (degree - 1), degree * (degree - 1)).cuda())
while True:
    optimizer = partial_optimizer(params=[params])

    t = tqdm(range(num_steps))
    for step in t:
        optimizer.zero_grad()
        adj_mat = model(params)
        hat = torch.matmul(adj_mat, adj_mat) + adj_mat
        loss_batch = F.mse_loss(hat, target, reduction='none').mean(dim=(1, 2))
        loss = loss_batch.sum()
        loss.backward()
        optimizer.step()
        t.set_postfix({'loss': loss_batch.min().item()})

    adj_mat = torch.round(model(params).detach())
    hat = torch.matmul(adj_mat, adj_mat) + adj_mat
    min_index = torch.argmin((hat - target).abs().sum(dim=(1, 2)))
    #print(adj_mat[min_index].to(torch.int8))
    #print((hat - target)[min_index].abs().sum().item())
    if (hat - target)[min_index].abs().sum().item() == 0:
        print("Success!")
        # save the adjacency matrix
        torch.save(adj_mat[min_index].to(torch.int8).cpu(), f'moore_degree{degree}_adj_mat.pt')
        break
    
    # flip sign of 10% of the parameters
    with torch.no_grad():
        num_params = params.numel()
        num_flip = num_params // 10
        indices = torch.randperm(num_params)[:num_flip]
        flat_params = params.view(-1)
        flat_params[indices] = -flat_params[indices]
        params.data = flat_params.view_as(params)
        params.data = params.data / params.data.std()  # re-normalize
