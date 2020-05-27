import torch
from torch import nn, optim
from torch.optim import lr_scheduler

model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, mode='min', patience=1, verbose=True)


# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                            factor=0.1,
#                                            patience=1, verbose=True,
#                                            min_lr=1e-5)

ndcg = [0.54,0.55,0.54,0.54,0.54,0.54,0.54,0.55,0.56]

for i in range(len(ndcg)):
    scheduler.step(-1*ndcg[i])
    print('Epoch ', i, ndcg[i])

    # scheduler.step(val_loss)
