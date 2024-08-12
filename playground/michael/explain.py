# %%
# Imports
import requests


# %%
layer = 0
feature = 0

res = requests.get(f'https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-res-jb/{feature}').json()


# %%
res['activations']