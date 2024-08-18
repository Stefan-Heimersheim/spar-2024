# %%
# Imports
import requests
from tqdm import trange
import pickle

# %%
model_name='gpt2-small'
sae_name='res-jb'
n_layers = 12
d_sae = 24576


# %%
# Download explanations per layer
explanations = [['' for _ in range(d_sae)] for _ in range(n_layers)]

for layer in trange(n_layers):
    res = requests.post('https://www.neuronpedia.org/api/explanation/export', json={'modelId': model_name, 'saeId': f'{layer}-{sae_name}'})

    for explanation in res.json()['explanations']:
        feature, description = int(explanation['index']), explanation['description']
        explanations[layer][feature] = description


# %%
# Save all explanations
with open('explanations.pkl', 'wb') as file:
    pickle.dump(explanations, file)
