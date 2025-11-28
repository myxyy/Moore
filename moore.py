import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

torch.set_printoptions(precision=1, edgeitems=1000, linewidth=1000)

device = torch.device(f"cuda:{args.gpu}")

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


degree = 57
batch_size = 1
num_steps = 50000
#degree = 7
#batch_size = 32
#num_steps = 50000

partial_optimizer = partial(torch.optim.Adam, lr=4e-1)

def moore_adj_mat(params: torch.Tensor, degree: int, mask: torch.Tensor) -> torch.Tensor:
    adj_mat = F.sigmoid(params)
    adj_mat = F.pad(adj_mat, (0, 1 + degree, 0, 1 + degree), value=1.0).triu()
    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat * mask
 
mask = moore_mask(degree).detach().to(device)
target = moore_target(degree).detach().to(device)

class MooreModel(nn.Module):
    def __init__(self, degree: int):
        super(MooreModel, self).__init__()
        self.degree = degree
        self.mask = moore_mask(degree).detach().to(device)
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        adj_mat = F.sigmoid(params)
        adj_mat = F.pad(adj_mat, (0, 1 + self.degree, 0, 1 + self.degree), value=1.0).triu()
        adj_mat = adj_mat + adj_mat.transpose(-2, -1)
        adj_mat = adj_mat * self.mask
        return adj_mat

model = MooreModel(degree).to(device)
target = target[None,:,:].expand(batch_size, -1, -1).to(device)
size = 1 + degree + (degree - 1) * degree


while True:
    params = torch.randn(batch_size, degree * (degree - 1), degree * (degree - 1)).to(device)
    params = nn.Parameter(params)
    optimizer = partial_optimizer(params=[params])
    t = tqdm(range(num_steps))
    for step in t:
        optimizer.zero_grad()
        #noise = torch.randn_like(params) / size
        #params.data.add_(noise)
        adj_mat_hat = model(params)
        target_hat = torch.matmul(adj_mat_hat, adj_mat_hat) + adj_mat_hat
        mse = F.mse_loss(target_hat, target, reduction='none')

        mse_diagonal = mse.diagonal(dim1=1, dim2=2)
        loss_diagonal = mse_diagonal.sum(dim=1) / size

        eye = torch.eye(size).to(mse.device)
        mse_without_diag = mse * (1 - eye)[None, :, :]
        loss_without_diag = mse_without_diag.sum(dim=(1, 2)) / (size * (size - 1))

        loss_batch = loss_diagonal * 1e-2 + loss_without_diag

        loss_batch_grad = torch.ones_like(loss_batch)
        loss_batch.backward(gradient=loss_batch_grad)
        optimizer.step()

        with torch.no_grad():
            min_loss = loss_batch.min().item()
            min_index = torch.argmin(loss_batch).item()
            min_param = params[min_index]
            best_adj_mat = model(min_param[None, :, :]).squeeze(0)
            best_target_hat = torch.matmul(best_adj_mat, best_adj_mat) + best_adj_mat
            diagonal = best_target_hat.diagonal()
            eye = torch.eye(best_target_hat.size(0)).to(best_target_hat.device)
            j = best_target_hat * (1 - eye) + eye

        t.set_postfix({'min_loss': f'{min_loss:.3f}', 'diag_mean': f'{diagonal.mean().item():.3f}', 'diag_std': f'{diagonal.std().item():.3f}', 'j_mean': f'{j.mean().item():.3f}', 'j_std': f'{j.std().item():.3f}'})

    adj_mat_round = torch.round(model(params).detach())
    hat_round = torch.matmul(adj_mat_round, adj_mat_round) + adj_mat_round
    min_index = torch.argmin((hat_round - target).abs().sum(dim=(1, 2)))
    if (hat_round - target)[min_index].abs().sum().item() == 0:
        print("Success!")
        # save the adjacency matrix
        torch.save(adj_mat_round[min_index].to(torch.int8).cpu(), f'moore_degree{degree}_adj_mat.pt')
        break
