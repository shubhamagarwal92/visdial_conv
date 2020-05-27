import torch


# x = torch.randn(1, 5)
x = torch.tensor([2,3,4,5,0,0])
mask = torch.eq(x, 0)

print(x)
print(mask)
scores = torch.tensor([2.4,3.5,4,5,6,7])

masked_x = scores.masked_fill(mask, 0)
print(masked_x)

# x.masked_fill_(mask, float('-inf'))
