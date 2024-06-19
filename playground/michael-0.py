# %%
from sae_lens import SAE

# %%
sae, cfg_dict, sparsity = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", device='cpu')