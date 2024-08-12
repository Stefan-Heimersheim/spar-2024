# %%
# Imports
import numpy as np


# %%
measure_name = 'cosine_similarity'
sae_name = 'res_jb_sae'

similarities = np.load(f'../../artefacts/similarity_measures/{measure_name}/{sae_name}_feature_similarity_{measure_name}_0.4.npz')['arr_0']


# %%
[(similarities[i].max(axis=1) > 0).sum() for i in range(11)]