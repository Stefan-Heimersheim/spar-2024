# %%
# Cosine similarities on gpt-3.5-turbo explanations
print('Loading explanations...')
with open('../../artefacts/explanations/res_jb_sae_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)

print('Loading embedder...')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print('Computing embeddings...')
embeddings = np.array([embedder.encode(layer_explanations) for layer_explanations in tqdm(explanations)])

similarities = np.array([cosine_similarity(embeddings[layer], embeddings[layer + 1]) for layer in trange(n_layers - 1)])

print('Computing similarities...')
forward_fig, backward_fig = similarity_pass_through(similarities)

print('Plotting...')
forward_fig.update_layout(title=f'Number of forward pass-through features (explanation similarity)')
backward_fig.update_layout(title=f'Number of backward pass-through features (explanation similarity)')

forward_fig.show()
backward_fig.show()




# %%
# Analyse some disappearing and appearing features to see if they are really not passed through
bound = 0.1

dead_features = (np.load(f'../../artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz')['arr_0'] == 0)[:-1]
disappearing_features = np.where((similarities.max(axis=2) < bound) & ~dead_features)
len(disappearing_features[0])

# %%
samples = np.random.choice(len(disappearing_features[0]), size=5, replace=False)


# %%
with open('../../artefacts/explanations/res_jb_sae_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)

for sample in samples:
    layer, feature = disappearing_features[0][sample], disappearing_features[1][sample]

    print(f'Disappearing feature {layer}_{feature} ({explanations[layer][feature]}):')

    top_5_downstream_neighbors = np.argpartition(similarities[layer, feature], -5)[-5:]

    for feature_ in top_5_downstream_neighbors:
        print(f'- {layer+1}_{feature_} ({explanations[layer+1][feature_]}) has {measure_name} {similarities[layer, feature, feature_]}')

    print('\n')


# %%
max_activations = np.load(f'../../artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz')['arr_0']

max_activations[8, 3615]


# %%
np.sort(similarities[8, 3615])