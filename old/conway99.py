import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import argparse

torch.set_printoptions(precision=1, edgeitems=1000, linewidth=1000)

#size = 99
#degree = 14
size = 9
degree = 4
#size = 243
#degree = 22

conway99_target = torch.eye(size) * (degree - 2) + torch.ones(size, size) * 2

batch_size = 32
device = torch.device("cuda:0")

class Conway99Model(nn.Module):
    def __init__(self, batch_size: int):
        super().__init__()
        self.params = nn.Parameter(torch.randn(batch_size, size, size))

    def forward(self) -> torch.Tensor:
        adj_mat = F.sigmoid(self.params).triu(1)
        adj_mat = adj_mat + adj_mat.transpose(-2, -1)
        return adj_mat

lr = 0.1
partial_optimizer = partial(torch.optim.Adam, lr=lr)

model = Conway99Model(batch_size).to(device)
target = conway99_target[None,:,:].expand(batch_size, -1, -1).to(device)

optimizer = partial_optimizer(model.parameters())
check_interval = 1000

eye = torch.eye(size).to(device)[None, :, :].expand(batch_size, -1, -1)
inverse_eye = (1 - eye).to(device)
ones = torch.ones(size).to(device)[None, :].expand(batch_size, -1)
#print(inverse_eye)

num_steps = 10000000
t = tqdm(range(num_steps), dynamic_ncols=True)
for step in t:
    optimizer.zero_grad()

    adj_mat_hat = model()
    #print(adj_mat_hat)
    target_hat = torch.matmul(adj_mat_hat, adj_mat_hat) + adj_mat_hat
    mse = F.mse_loss(target_hat, target, reduction='none')
    loss_batch_diagonal = (mse * eye).sum(dim=(1, 2)) / size
    loss_batch_off_diagonal = (mse * inverse_eye).sum(dim=(1, 2)) / (size * (size - 1))
    loss_batch_column_regularity = F.mse_loss(target_hat.sum(dim=2) / degree, ones, reduction='none').mean(dim=1)
    loss_batch_row_regularity = F.mse_loss(target_hat.sum(dim=1) / degree, ones, reduction='none').mean(dim=1)
    loss_batch = loss_batch_off_diagonal# + loss_batch_diagonal# + loss_batch_column_regularity + loss_batch_row_regularity
    loss_batch_grad = torch.ones_like(loss_batch)
    loss_batch.backward(gradient=loss_batch_grad)
    optimizer.step()

    min_loss = loss_batch.min().item()

    t.postfix = {'loss': f"{min_loss:.6f}"}

    if step % check_interval == 0:
        with torch.no_grad():
            adj_mat_round = torch.round(model().detach())
            hat_round = torch.matmul(adj_mat_round, adj_mat_round) + adj_mat_round
            min_index = torch.argmin(torch.round(hat_round - target).abs().sum(dim=(1, 2)))
            #print(adj_mat_round[min_index].to(torch.int8))
            #print(torch.round(hat_round)[min_index].to(torch.int8))
            if torch.round(hat_round - target)[min_index].abs().sum().item() == 0:
                print("Success!")
                # save the adjacency matrix
                torch.save(adj_mat_round[min_index].to(torch.int8).cpu(), f'conway{size}_adj_mat.pt')
                is_success = True
                break
