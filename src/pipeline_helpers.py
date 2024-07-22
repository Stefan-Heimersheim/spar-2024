# %%
# Imports
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
import einops
from torch.utils.data import DataLoader
from tqdm import tqdm

from similarity_measures import Aggregator



def load_model_and_saes(model_name, sae_name, hook_name, device='cuda') -> tuple[HookedTransformer, list[SAE]]:
    model = HookedTransformer.from_pretrained(model_name, device=device)

    saes = []
    for layer in tqdm(list(range(model.cfg.n_layers))):
        sae, _, _ = SAE.from_pretrained(
            release=sae_name,
            sae_id=f"blocks.{layer}.{hook_name}",
            device=device,
        )
        sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
        saes.append(sae)

    return model, saes


def load_data(model, sae, dataset_name, number_of_batches=None, batch_size=32):
    context_size = sae.cfg.context_size
    prepend_bos = sae.cfg.prepend_bos
    
    dataset = load_dataset(path=dataset_name, split="train", streaming=False)

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=context_size,
        add_bos_token=prepend_bos
    )

    # Cut off to avoid a partial batch at the end
    tokens = token_dataset["tokens"][:((len(token_dataset) // batch_size) * batch_size)]

    if number_of_batches is None:
        return tokens
    else:
        return tokens[:(number_of_batches * batch_size)]

def get_device() -> str:
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def load_data2(
    model, context_size=128, prepend_bos=True, dataset_name='NeelNanda/pile-10k', number_of_batches=None, batch_size=32
):
    """
    
    """
    dataset = load_dataset(path=dataset_name, split="train", streaming=False)

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=context_size,
        add_bos_token=prepend_bos
    )

    # Cut off to avoid a partial batch at the end
    tokens = token_dataset["tokens"][:((len(token_dataset) // batch_size) * batch_size)]

    if number_of_batches is None:
        return tokens
    else:
        return tokens[:(number_of_batches * batch_size)]

def run_with_aggregator(model, saes, hook_name, tokens, aggregator : Aggregator, batch_size=32):
    data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

    context_size = saes[0].cfg.context_size
    d_sae = saes[0].cfg.d_sae
    sae_activations = torch.empty(model.cfg.n_layers, d_sae, batch_size * context_size)


    def retrieval_hook(activations, hook):
        layer = hook.layer()

        sae_activations[layer] = einops.rearrange(
            # Get SAE activations
            saes[layer].encode(activations), "batch seq features -> features (batch seq)"
        )


    model.add_hook(lambda name: name.endswith(f".{hook_name}"), retrieval_hook)

    with torch.no_grad():
        for batch_tokens in tqdm(data_loader):
            model.run_with_hooks(batch_tokens)

            # Now we can use sae_activations
            aggregator.process(sae_activations)

        return aggregator.finalize()
