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
#batch_size = 4
#num_steps = 25000
degree = 7
batch_size = 8
num_steps = 10000

mutation_rate = 0.5

preserve_gene = True


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

scale = 1e-3

gene = torch.randn(degree * (degree - 1), degree * (degree - 1)).cuda() * scale
lowest_loss = float('inf')

while True:
    pool = gene[None,:,:].expand(batch_size, -1, -1).clone()
    num_params = pool.numel() // batch_size
    num_flip = int(num_params * mutation_rate)
    for i in range(batch_size):
        if preserve_gene and i == 0:
            continue
        rand = torch.rand_like(pool[i])
        flip = (rand < mutation_rate).float() * 2 - 1
        pool[i] = pool[i] * flip
        pool[i] = pool[i] / pool[i].std() * scale

    params = nn.Parameter(pool)
    optimizer = partial_optimizer(params=[params])

    t = tqdm(range(num_steps))
    for step in t:
        optimizer.zero_grad()
        adj_mat = model(params)
        hat = torch.matmul(adj_mat, adj_mat) + adj_mat
        loss_batch = F.mse_loss(hat, target, reduction='none').mean(dim=(1, 2))
        loss_batch_grad = torch.ones_like(loss_batch)
        loss_batch.backward(gradient=loss_batch_grad)
        optimizer.step()
        min_loss = loss_batch.min().item()
        second_min_loss = torch.topk(loss_batch, 2, largest=False).values[1].item()
        t.set_postfix({'min_loss': min_loss, 'second_min_loss': second_min_loss})

    adj_mat_round = torch.round(model(params).detach())
    hat_round = torch.matmul(adj_mat_round, adj_mat_round) + adj_mat_round
    min_index = torch.argmin((hat_round - target).abs().sum(dim=(1, 2)))
    #print(adj_mat_round[min_index].to(torch.int8))
    #print((hat_round - target)[min_index].abs().sum().item())
    if (hat_round - target)[min_index].abs().sum().item() == 0:
        print("Success!")
        # save the adjacency matrix
        torch.save(adj_mat[min_index].to(torch.int8).cpu(), f'moore_degree{degree}_adj_mat.pt')
        break

    adj_mat = model(params).detach()
    hat = torch.matmul(adj_mat, adj_mat) + adj_mat
    loss_batch = F.mse_loss(hat, target, reduction='none').mean(dim=(1, 2))
    loss_min_index = torch.argmin(loss_batch)
    if loss_batch[loss_min_index].item() < lowest_loss:
        lowest_loss = loss_batch[loss_min_index].item()
        gene = params[loss_min_index].detach().clone()

