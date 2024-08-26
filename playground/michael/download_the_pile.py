# %%
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import json
from tqdm import tqdm


# %%
device = 'cuda:0'

# %%
def save_subset_of_pile(output_file, num_tokens=100_000_000):
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    tokenizer = model.tokenizer
    
    token_count = 0
    saved_data = []

    progress_bar = tqdm(dataset, total=num_tokens)
    for i, example in progress_bar:
        tokens = tokenizer.encode(example['text'])
        token_count += len(tokens)
        saved_data.append(example['text'])

        progress_bar.n = token_count
        
        if token_count >= num_tokens:
            break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(saved_data, f)
    
    print(f"Saved approximately {token_count} tokens to {output_file}")


# %%
save_subset_of_pile("../../artefacts/pile_subset.json")