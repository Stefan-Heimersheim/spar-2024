# %%
import torch
import einops


# %%
x = torch.tensor([[0, 1, 1], [1, 0, 0]])
y = torch.tensor([[1, 1, 1], [1, 1, 0]])

intersection = einops.einsum(x, y, 'f1 t, f2 t -> f1 f2')
sum_x = x.sum(dim=-1)


# %%
intersection / sum_x


# %%
active = (torch.rand(2, 100, 1000) > 0.9).float()

# %%
result = einops.einsum(active[0], active[1], 'f1 t, f2 t -> f1 f2') / active[0].sum(dim=-1)
result.shape
