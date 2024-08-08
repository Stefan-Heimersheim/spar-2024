# %%
from abc import ABC
import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

# %%
# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
device = next(model.parameters()).device

prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)

layer_index = 6
location = "resid_post_mlp"

transformer_lens_loc = {
    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
}[location]

# %%
with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)

# %%
input_tensor = activation_cache[transformer_lens_loc]

input_tensor_ln = input_tensor

with torch.no_grad():
    latent_activations, info = autoencoder.encode(input_tensor_ln)
    reconstructed_activations = autoencoder.decode(latent_activations, info)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
print(location, normalized_mse)

# %%


# %%
class SAE(ABC):
    def encode(self, activations):
        pass

    @property
    def d_sae(self, ):
        pass

    @property
    def context_size(self, ):
        pass

    @property
    def prepend_bos(self, ):
        pass


class SAELens_SAE(SAE):
    def __init__(self, sae) -> None:
        super().__init__()

        self.sae = sae

    def encode(self, activations):
        return self.sae.encode(activations)

    @property
    def d_sae(self):
        return self.sae.cfg.d_sae
    
    @property
    def context_size(self):
        return self.sae.cfg.context_size
    
    @property
    def prepend_bos(self):
        return self.sae.cfg.prepend_bos


class OpenAI_SAE(SAE):
    def __init__(self, location, layer) -> None:
        with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer), mode="rb") as f:
            state_dict = torch.load(f)
            self.autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
            self.autoencoder.to(device)

        self.d_sae = 32768

    def encode(self, activations):
        return self.autoencoder.encode(activations)      
      