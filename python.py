import torch
import torch.nn as nn

model  = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

X = torch.tensor([[0.,0.], [0.,1.], [1., 0.],[1.,1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])
