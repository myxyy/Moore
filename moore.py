import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
parser.add_argument("--degree", type=int, default=57, choices=[7, 57], help="Degree of the Moore graph (default: 57)")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training (default: 1)")
parser.add_argument("--num_steps", type=int, default=1000000, help="Number of optimization steps (default: 1000000)")
parser.add_argument("--lr", type=float, default=0.4, help="Learning rate for the optimizer (default: 0.4)")
parser.add_argument("--diagonal_weight", type=float, default=1e-2, help="Weight for the diagonal loss component (default: 1e-2)")
parser.add_argument("--check_interval", type=int, default=1000, help="Interval for checking progress (default: 1000)")
parser.add_argument("--noise_scale", type=float, default=0.1, help="Scale of the noise added to the target (default: 0.1)")
parser.add_argument("--regularity_weight", type=float, default=100.0, help="Weight for the regularity loss component (default: 100.0)")
args = parser.parse_args()

degree = args.degree
batch_size = args.batch_size
num_steps = args.num_steps
lr = args.lr
diagonal_weight = args.diagonal_weight
check_interval = args.check_interval
noise_scale = args.noise_scale
regularity_weight = args.regularity_weight

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

partial_optimizer = partial(torch.optim.Adam, lr=lr)

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

is_success = False
while not is_success:
    params = torch.randn(batch_size, degree * (degree - 1), degree * (degree - 1)).to(device)
    params = nn.Parameter(params)
    optimizer = partial_optimizer(params=[params])
    t = tqdm(range(num_steps), dynamic_ncols=True)
    for step in t:
        optimizer.zero_grad()
        #noise = torch.randn_like(params) / size
        #params.data.add_(noise)
        adj_mat_hat = model(params)
        target_hat = torch.matmul(adj_mat_hat, adj_mat_hat) + adj_mat_hat
        target_with_noise = target + torch.randn_like(target_hat) * noise_scale
        mse = F.mse_loss(target_hat, target_with_noise, reduction='none')

        mse_diagonal = mse.diagonal(dim1=1, dim2=2)
        loss_diagonal = mse_diagonal.sum(dim=1) / size

        eye = torch.eye(size).to(mse.device)
        mse_without_diag = mse * (1 - eye)[None, :, :]
        loss_without_diag = mse_without_diag.sum(dim=(1, 2)) / (size * (size - 1))


        if regularity_weight > 0:
            adj_mat_hat_param_part = adj_mat_hat[:, :degree * (degree - 1), :degree * (degree - 1)] + torch.eye(degree * (degree - 1)).to(adj_mat_hat.device)[None, :, :]
            adj_mat_hat_param_part_reshape = adj_mat_hat_param_part.reshape(batch_size, degree * (degree - 1), degree, degree - 1)
            col_sum = adj_mat_hat_param_part_reshape.sum(dim=3)
            regularity_loss = F.mse_loss(col_sum, torch.ones_like(col_sum) + torch.randn_like(col_sum) * noise_scale, reduction='none').mean(dim=(1,2))
        else:
            regularity_loss = 0
        
        loss_batch = loss_diagonal * diagonal_weight + loss_without_diag + regularity_loss * regularity_weight

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

        t.set_postfix({
            'min_loss': f'{min_loss:.3f}',
            'diag_mean': f'{diagonal.mean().item():.3f}',
            'j_mean': f'{j.mean().item():.3f}'
        })

        if step % check_interval == 0:
            adj_mat_round = torch.round(model(params).detach())
            hat_round = torch.matmul(adj_mat_round, adj_mat_round) + adj_mat_round
            min_index = torch.argmin((hat_round - target).abs().sum(dim=(1, 2)))
            if (hat_round - target)[min_index].abs().sum().item() == 0:
                print("Success!")
                # save the adjacency matrix
                torch.save(adj_mat_round[min_index].to(torch.int8).cpu(), f'moore_degree{degree}_adj_mat.pt')
                is_success = True
                break
