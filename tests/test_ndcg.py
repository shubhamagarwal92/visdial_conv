import torch


# ranks = [54, 67, 59, 46, 2, 1, 100]
ranks = [5, 6, 4, 3, 2, 1, 7]
predicted_ranks = torch.tensor(ranks).float()
predicted_ranks = predicted_ranks.unsqueeze(0)
print(predicted_ranks.size())
new_ranks, rankings = torch.sort(predicted_ranks, dim=-1)

print("Original: ", predicted_ranks)
print("Rankings", rankings)
print("New ranks", new_ranks)
