# %%
import pickle

# %%
# Unpickle the object
with open('/Users/benlerner/work/spar-2024/artefacts/graphs/res_jb_sae_feature_similarity_necessity_10M_relative_activation_0.2_threshold_0.99.pkl', 'rb') as file:
    graph = pickle.load(file)

# Inspect the object
print(f"Type of the object: {type(graph)}")
print(f"Object content: {graph}")

# If it's a complex object, you might want to explore its attributes
if hasattr(graph, '__dict__'):
    print("Object attributes:")
    for attr, value in graph.__dict__.items():
        print(f"  {attr}: {value}")
# %%
dir(graph)

# %%
