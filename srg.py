import torch
import torch.nn as nn
import math
from functools import partial
from tqdm import tqdm
import torch.nn.functional as F
import sys
import argparse

named_parameters = {
    "conway99": (14, 1, 2),
    "hoffman_singleton": (7, 0, 1),
    "moore57": (57, 0, 1),
    "petersen": (3, 0, 1),
    "gewirtz": (10, 0, 2),
    "clebsch": (5, 0, 2),
    "shrikhande": (6, 2, 2),
    "benchmark0": (16, 9, 12),
}

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: %(default)d)")

parser.add_argument("--name", type=str, default=None, choices=[None] + list(named_parameters.keys()), help="Name of the SRG to find (default: %(default)s)")
parser.add_argument("--degree", type=int, default=14, help="Degree of SRG (default: %(default)d)")
parser.add_argument("--lmd", type=int, default=1, help="SRG parameter lambda (default: %(default)d)")
parser.add_argument("--mu", type=int, default=2, help="SRG parameter mu (default: %(default)d)")

parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training (default: %(default)d)")
parser.add_argument("--check_interval", type=int, default=10, help="Interval for checking progress (default: %(default)d)")

parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer (default: %(default)f)")
parser.add_argument("--noise_scale", type=float, default=0.0, help="Scale of the noise added to the adjacency matrix loss (default: %(default)f)")

parser.add_argument("--orthogonal_weight", type=float, default=10.0, help="Weight for the orthogonal loss component (default: %(default)f)")
parser.add_argument("--qjq_weight", type=float, default=10.0, help="Weight for the qjq loss component (default: %(default)f)")
parser.add_argument("--orthogonal_diagonal_weight", type=float, default=10.0, help="Weight for the orthogonal diagonal loss component (default: %(default)f)")
parser.add_argument("--qjq_diagonal_weight", type=float, default=1, help="Weight for the qjq diagonal loss component (default: %(default)f)")
parser.add_argument("--adj_diagonal_weight", type=float, default=1e-3, help="Weight for the adj diagonal loss component (default: %(default)f)")

parser.add_argument("--range_penalty_weight", type=float, default=100.0, help="Weight for the range penalty loss component (default: %(default)f)")
parser.add_argument("--regularity_weight", type=float, default=0.1, help="Weight for the regularity loss component (default: %(default)f)")
parser.add_argument("--zero_diag_weight", type=float, default=10.0, help="Weight for the zero diagonal loss component (default: %(default)f)")
args = parser.parse_args()

batch_size = args.batch_size
lr = args.lr
orthogonal_weight = args.orthogonal_weight
qjq_weight = args.qjq_weight
range_penalty_weight = args.range_penalty_weight
k = args.degree
l = args.lmd
m = args.mu
name = args.name
noise_scale = args.noise_scale
check_interval = args.check_interval
orthogonal_diagonal_weight = args.orthogonal_diagonal_weight
qjq_diagonal_weight = args.qjq_diagonal_weight
adj_diagonal_weight = args.adj_diagonal_weight
regularity_weight = args.regularity_weight
zero_diag_weight = args.zero_diag_weight

print(f"lr: {lr}")
print(f"orthogonal_weight: {orthogonal_weight}")
print(f"qjq_weight: {qjq_weight}")
print(f"range_penalty_weight: {range_penalty_weight}")
print(f"noise_scale: {noise_scale}")
print(f"orthogonal_diagonal_weight: {orthogonal_diagonal_weight}")
print(f"qjq_diagonal_weight: {qjq_diagonal_weight}")
print(f"adj_diagonal_weight: {adj_diagonal_weight}")
print(f"regularity_weight: {regularity_weight}")
print(f"zero_diag_weight: {zero_diag_weight}")
print(f"batch_size: {batch_size}")

torch.set_printoptions(precision=1, edgeitems=1000, linewidth=1000)

if name is not None:
    k, l, m = named_parameters[args.name]

if k * (k - l - 1) % m != 0:
    raise ValueError("SRG parameters are invalid: k*(k-l-1) is not divisible by m.")
v = k * (k - l - 1) // m + k + 1
print(f"v: {v}, k: {k}, l: {l}, m: {m}")

d = (l - m) * (l - m) + 4 * (k - m)
sqrt_d = math.sqrt(d)
if round(sqrt_d) ** 2 != d and 2 * k + (v - 1) * (l - m) != 0:
    raise ValueError("Eigenvalue multiplicities are not integers.")

# now round(sqrt_d) ** 2 == d or 2 * k + (v - 1) * (l - m) == 0

eigenvalue1 = (l - m + sqrt_d) / 2
eigenvalue2 = (l - m - sqrt_d) / 2

e = (2 * k + (v - 1) * (l - m))
if round(sqrt_d) ** 2 == d and e % round(sqrt_d) != 0:
    raise ValueError("Eigenvalue multiplicities are not integers.")

f = e // round(sqrt_d)
if ((v - 1) - f) % 2 != 0:
    raise ValueError("Eigenvalue multiplicities are not integers.")

multiplicity1 = ( (v - 1) - f ) // 2
multiplicity2 = ( (v - 1) + f ) // 2

eigenvalue_list = [k, eigenvalue1, eigenvalue2]
multiplicity_list = [1, multiplicity1, multiplicity2]
#print("Eigenvalues and their multiplicities:")
#for ev, mult in zip(eigenvalue_list, multiplicity_list):
#    print(f"Eigenvalue: {ev}, Multiplicity: {mult}")
print(f"Eigenvalues: {eigenvalue_list}")
print(f"Multiplicities: {multiplicity_list}")

def separate_diagonal_loss(loss_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    diag_elements = torch.diagonal(loss_mat, dim1=-2, dim2=-1)
    off_diag_elements = loss_mat - torch.diag_embed(diag_elements)
    diag_loss = diag_elements.mean(dim=1)
    off_diag_loss = off_diag_elements.sum(dim=(1,2)) / (loss_mat.size(1) * (loss_mat.size(2) - 1))
    return diag_loss, off_diag_loss

device = torch.device(f"cuda:{args.gpu}")

diagonal = []
for ev, mult in zip(eigenvalue_list, multiplicity_list):
    diagonal.extend([ev] * mult)
diagonal_tensor = torch.tensor(diagonal, dtype=torch.float32).to(device)
#print("Diagonal tensor:", diagonal_tensor)
#print("Test:", diagonal_tensor * diagonal_tensor + (m - l) * diagonal_tensor + (m - k) * torch.ones_like(diagonal_tensor))

#sys.exit(0)

q = nn.Parameter(torch.randn(batch_size, v, v, device=device) * v ** -0.5)
partial_optimizer = partial(torch.optim.AdamW, lr=lr)
optimizer = partial_optimizer([q])
eyes = torch.eye(v).to(device)[None, :, :].expand(batch_size, -1, -1).to(device)

mqjq_target_vector = diagonal_tensor * diagonal_tensor + (m - l) * diagonal_tensor + (m - k) * torch.ones_like(diagonal_tensor)
mqjq_target = torch.diag_embed(mqjq_target_vector)[None, :, :].expand(batch_size, -1, -1).to(device)

min_srg_test = 65536

step = 0
is_info_printed = False

while True:
    try:
        optimizer.zero_grad()
        d = torch.diag_embed(diagonal_tensor)

        orhogonal_loss_raw = F.mse_loss(torch.matmul(q.transpose(-2, -1), q), eyes, reduction='none')
        orhogonal_loss_diag, orhogonal_loss_off_diag = separate_diagonal_loss(orhogonal_loss_raw)
        orhogonal_loss = orhogonal_loss_diag * orthogonal_diagonal_weight + orhogonal_loss_off_diag

        qjq = torch.matmul(q.transpose(-2, -1), torch.matmul(torch.ones_like(q), q))
        qjq_loss_raw = F.mse_loss(m * qjq, mqjq_target, reduction='none')
        qjq_diag_loss, qjq_off_diag_loss = separate_diagonal_loss(qjq_loss_raw)
        qjq_loss = qjq_diag_loss * qjq_diagonal_weight + qjq_off_diag_loss

        regularity_loss = F.mse_loss(torch.matmul(d, qjq), torch.matmul(qjq, d), reduction='none').mean(dim=(1,2))

        adj_mat_hat = torch.matmul(q, torch.matmul(d, q.transpose(-2, -1)))
        adj_lhs = torch.matmul(adj_mat_hat, adj_mat_hat) + (m - l) * adj_mat_hat + (m - k) * eyes - m * torch.ones_like(adj_mat_hat)
        noise = torch.randn_like(adj_lhs) * noise_scale * adj_lhs.std(dim=(-2, -1), keepdim=True).detach()
        adj_loss_raw = F.mse_loss(adj_lhs, noise, reduction='none')
        adj_diagonal_loss, adj_off_diagonal_loss = separate_diagonal_loss(adj_loss_raw)
        adj_loss = adj_diagonal_loss * adj_diagonal_weight + adj_off_diagonal_loss

        zero_diag_loss = torch.diagonal(adj_mat_hat, dim1=-2, dim2=-1).pow(2).mean(dim=1)

        over_one_penalty = F.relu(adj_mat_hat - 1).mean(dim=(1,2))
        under_zero_penalty = F.relu(-adj_mat_hat).mean(dim=(1,2))
        range_penalty = over_one_penalty + under_zero_penalty

        loss_batch =\
            orhogonal_loss * orthogonal_weight + \
            qjq_loss * qjq_weight + \
            range_penalty * range_penalty_weight + \
            regularity_loss * regularity_weight + \
            zero_diag_loss * zero_diag_weight + \
            adj_loss
        loss_batch_grad = torch.ones_like(loss_batch)
        loss_batch.backward(gradient=loss_batch_grad)
        optimizer.step()

        loss_min_index = torch.argmin(loss_batch)

        if step % check_interval == 0:
            with torch.no_grad():
                adj_mat_hat_for_check = torch.matmul(q, torch.matmul(torch.diag_embed(diagonal_tensor), q.transpose(-2, -1)))
                round_adj_mat_hat = torch.round(torch.clamp(adj_mat_hat_for_check, 0, 1))
                srg_test = torch.matmul(round_adj_mat_hat, round_adj_mat_hat) + (m - l) * round_adj_mat_hat + (m - k) * torch.eye(v).to(device)[None, :, :].expand(batch_size, -1, -1) - m * torch.ones_like(round_adj_mat_hat)
                srg_test_batch = srg_test.abs().sum(dim=(1,2))
                min_index = torch.argmin(srg_test_batch)
                min_srg_test = srg_test_batch[min_index].item()

        info = \
            f'step: {step}                \n'\
            f'min_srg_test: {min_srg_test}                \n'\
            f'min_loss: {loss_batch[loss_min_index].item():.4f}                \n'\
            f'orthogonal_loss: {orhogonal_loss[loss_min_index].item():.4f}                \n'\
            f'qjq_loss: {qjq_loss[loss_min_index].item():.4f}                \n'\
            f'range_penalty: {range_penalty[loss_min_index].item():.4f}                \n'\
            f'regularity_loss: {regularity_loss[loss_min_index].item():.4f}                \n'\
            f'zero_diag_loss: {zero_diag_loss[loss_min_index].item():.4f}                \n'\
            f'adj_loss: {adj_loss[loss_min_index].item():.4f}                \n'\

        if is_info_printed:
            info = '\033[9A' + info
        print(info, end='', flush=True)
        is_info_printed = True

        if min_srg_test == 0:
            print("Found SRG!")
            #print(round_adj_mat_hat[min_index].to(torch.int8))
            path = f"srg_v{v}_k{k}_l{l}_m{m}.pt"
            torch.save(round_adj_mat_hat[min_index].to(torch.int8).cpu(), path)
            print(f"Saved to {path}")
            break

        step += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
        break

