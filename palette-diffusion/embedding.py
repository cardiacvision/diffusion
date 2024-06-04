#%%
import math
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity, pairwise_euclidean_distance
import numpy as np

def gamma_embedding(gammas, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param gammas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

param1 = torch.tensor([0.00025, 0.0005, 0.001, 0.00175, 0.0275])
param2 = torch.tensor([0.0035, 0.0285, 0.0535, 0.0785, 0.1035])
param3 = torch.linspace(0, 1, 4000//10)

# param1_enc = param1
# param2_enc = param2
param1_enc = -torch.log(param1)
param2_enc = -torch.log(param2)

max_period = 1
param3_emb = gamma_embedding(param3, 128, max_period=max_period)
max_period = 10
param1_emb = gamma_embedding(param1_enc, 128, max_period=max_period)
max_period = 10
param2_emb = gamma_embedding(param2_enc, 128, max_period=max_period)


#%%
cosine_sim1 = pairwise_cosine_similarity(param1_emb, zero_diagonal=False)
cosine_sim2 = pairwise_cosine_similarity(param2_emb, zero_diagonal=False)
cosine_sim12 = pairwise_cosine_similarity(param1_emb, param2_emb, zero_diagonal=False)
cosine_sim3 = pairwise_cosine_similarity(param3_emb, zero_diagonal=False)
#%%

param1 = [0.00025, 0.0005, 0.001, 0.00175, 0.0275]
param2 = [0.0035, 0.0285, 0.0535, 0.0785, 0.1035]
param3 = np.linspace(0, 1, 4000//10)

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
s = sns.heatmap(cosine_sim1.numpy(),
                xticklabels=param1, yticklabels=param1,
                ax=axs[0],
                vmin=-1, vmax=1, 
                cmap='vlag')
s.set_xlabel('param1')
s.set_ylabel('param1')

s = sns.heatmap(cosine_sim2.numpy(),
                xticklabels=param2, yticklabels=param2,
                ax=axs[1],
                vmin=-1, vmax=1, 
                cmap='vlag')
s.set_xlabel('param2')
s.set_ylabel('param2')

# s = sns.heatmap(cosine_sim12.numpy(),
#                 ax=axs[2],
#                 xticklabels=param1, yticklabels=param2,
#                 # vmin=-1, vmax=1, 
#                 cmap='vlag')
# s.set_xlabel('param1')
# s.set_ylabel('param2')

s = sns.heatmap(cosine_sim3.numpy(),
                ax=axs[2],
                # xticklabels=param3, yticklabels=param3,
                vmin=-1, vmax=1, 
                cmap='vlag')
s.set_xlabel('param3')
s.set_ylabel('param3')
plt.show()
# %%
