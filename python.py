import torch
import torch.nn as nn

model  = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

X = torch.tensor([[0.,0.], [0.,1.], [1., 0.],[1.,1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

loss_fn=nn.MSELoss()
opt=torch.optim.Adam(model.parameters(), lr=0.01)


for _ in range(1000) :
  opt.zero_grad()
  loss = loss_fn(model(X), y)
  loss.backward()
  opt.step()
