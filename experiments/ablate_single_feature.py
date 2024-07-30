## https://spar2024.slack.com/archives/C0794GNT8KS/p1720559352313959?thread_ts=1720550352.237039&cid=C0794GNT8KS

"""
Run ablation of feature in layer 1 and record all features in layer 2 at almost no additional cost -->
getting lots more ablation scores at once

1. implement for all layer 2 features
2. pick 3 random layer 1 features
3. collect ablation acts for each layer 1 feature
"""

# %%
from dataclasses import dataclass
import argparse
import torch as t
import torch
import typing
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
from functools import partial
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Loaded {device=}")


# %%
@dataclass
class AblationAggregator:
    num_batches: int
    batch_size: int
    model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device=device)

    def create_id_to_sae(self) -> typing.Dict[str, SAE]:
        print("Loading SAEs")
        sae_id_to_sae = {}
        for layer in tqdm(list(range(self.model.cfg.n_layers))):
           sae_id = f"blocks.{layer}.hook_resid_pre"
           sae, _, _ = SAE.from_pretrained(
               release="gpt2-small-res-jb",
               sae_id=sae_id,
               device=device
           )
           sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
           sae_id_to_sae[sae_id] = sae 
        return sae_id_to_sae

    # https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
    def __post_init__(self) -> None:
        """
        f2 = the second feature activation in the subsequent layer
        f2_diffs = difference between f2 when f1 (the feature in the previous layer) is ablated vs unablated
        masked_f2_diffs = f2_diffs, but only if unablated_f2 was non-zero
        """
        self.sae_id_to_sae = self.create_id_to_sae()
        # These hyperparameters are used to pre-process the data
        pre_0_sae_id = "blocks.0.hook_resid_pre"
        pre_0_sae = self.sae_id_to_sae[pre_0_sae_id]
        self.context_size = pre_0_sae.cfg.context_size
        self.prepend_bos = pre_0_sae.cfg.prepend_bos
        self.d_sae = pre_0_sae.cfg.d_sae

        self.sae_errors = t.empty(self.batch_size, self.context_size, self.model.cfg.d_model)
        self.second_layer_unablated_acts = t.empty(self.batch_size, self.context_size, self.d_sae)
        self.second_layer_ablated_acts = t.empty(self.batch_size, self.context_size, self.d_sae)

        # these get set later
        self.first_layer_idx = None
        self.feature_idx = None
        
        self.num_tokens_per_batch = self.batch_size*self.context_size
        self.sum_of_f2_diffs = t.zeros(self.d_sae).to(device)
        self.sum_of_squared_f2_diffs = t.zeros(self.d_sae).to(device)
        self.sum_of_squared_masked_f2_diffs= t.zeros(self.d_sae).to(device)
        self.n_total = 0
        self.masked_n = t.zeros(self.d_sae).to(device) # number of original activations that were > min_activation
        self.min_activation_tol = 1e-15 # TODO: should be diff?
        self.mean_diffs = t.zeros(self.d_sae).to(device)
        self.m2_diffs = t.zeros(self.d_sae).to(device)
        self.sum_unablated_f2 = t.zeros(self.d_sae).to(device)
        self.sum_ablated_f2 = t.zeros(self.d_sae).to(device)
        
        # pre-defining the aggregates so that we don't have issues with defining vars outside if __init__
        self.mse = None
        self.masked_mse = None
        self.variances = None
        self.std_devs = None
        self.masked_variances = None
        self.masked_stdevs = None

        # will only contain the means of the activation diffs in which the first layer's activation was > 0
        self.masked_means = t.zeros(self.d_sae).to(device)
        self.masked_m2 = t.zeros(self.d_sae).to(device)

    def _load_data(self) -> DataLoader:
        dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,  # type: ignore
            tokenizer=self.model.tokenizer,  # type: ignore
            streaming=True,
            max_length=self.context_size,
            add_bos_token=self.prepend_bos,
        )
        num_of_sentences = self.batch_size * self.num_batches
        tokens = token_dataset['tokens'][:num_of_sentences]
        return DataLoader(tokens, batch_size=self.batch_size, shuffle=False)

    def aggregate(
        self,
        first_layer_idx: int,
        feature_idx: int,
    ):
        self.first_layer_idx = first_layer_idx
        self.feature_idx= feature_idx
        data_loader = self._load_data()
        print("Collecting diff aggs")
        num_batches_left = self.num_batches
        with torch.no_grad():
            for batch_tokens in tqdm(data_loader):
                if num_batches_left == 0:
                    break
                self.model.reset_hooks()
                # collect the unablated activations
                self.model.run_with_hooks(
                    batch_tokens,
                    fwd_hooks=[
                        (
                            f"blocks.{self.first_layer_idx}.hook_resid_pre", 
                            self.save_error_terms,
                        ),
                        (
                            f"blocks.{self.first_layer_idx+1}.hook_resid_pre", 
                            partial(self.save_second_layer_acts, ablated=False),
                        )
                    ]
                )
                # collect the activations after ablation
                self.model.run_with_hooks(
                    batch_tokens,
                    fwd_hooks=[
                        (
                            f"blocks.{self.first_layer_idx}.hook_resid_pre", 
                            partial(self.ablate_and_reconstruct_with_errors, feature_idx=self.feature_idx)
                        ),
                        (
                            f"blocks.{self.first_layer_idx+1}.hook_resid_pre", 
                            partial(self.save_second_layer_acts, ablated=True)
                        )
                    ]
                )
                self._process_global_second_layer_acts()
                num_batches_left -= 1
        self._finalize()

    def save_error_terms(self, activations: t.Tensor, hook: HookPoint) -> t.Tensor:
        sae: SAE = self.sae_id_to_sae[hook.name]
        reconstructed_acts = sae(activations)
        self.sae_errors = activations - reconstructed_acts 
        return activations

    def save_second_layer_acts(self, activations: t.Tensor, hook: HookPoint, ablated: bool):
        sae: SAE = self.sae_id_to_sae[hook.name]
        sae_feats = sae.encode(activations)
        if ablated:
            self.second_layer_ablated_acts = sae_feats
        else:
            self.second_layer_unablated_acts = sae_feats
        return activations

    def ablate_and_reconstruct_with_errors(self, activations: t.Tensor, hook: HookPoint, feature_idx: int):
        sae: SAE = self.sae_id_to_sae[hook.name]
        sae_feats = sae.encode(activations)
        sae_feats[:,:,feature_idx] = 0
        return sae.decode(sae_feats) + self.sae_errors # we assume that the hooks are called in the correct order s.t. these errors correspond to the same tokens

    def _process_global_second_layer_acts(self):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        let a be the previous `self` aggregations and b be the new batch's aggregations
        and ab is the result of combining the previous aggregations with the batch aggs
        """
        curr_diffs = self.second_layer_unablated_acts - self.second_layer_ablated_acts
        # create the local vars to match the algorithm
        n_a = self.n_total
        n_b = self.num_tokens_per_batch
        n_ab = n_a + n_b
        mean_b = curr_diffs.mean(dim=(0,1))
        mean_a = self.mean_diffs
        delta = mean_b - mean_a
        mean_ab = mean_a + delta * (n_b / n_ab)
        m2_a = self.m2_diffs
        # M2 aggregates the squared distance from the mean
        m2_b = (mean_b - curr_diffs).pow(2).sum(dim=(0, 1))
        m2_ab = m2_a + m2_b + (delta.pow(2) * n_a * n_b / n_ab)
        
        # process only the activations where the first layer was active
        active_mask = self.second_layer_unablated_acts > self.min_activation_tol # TODO: do some sort of tolerance?
        masked_diffs = (curr_diffs * active_mask)
        masked_n_a = self.masked_n
        masked_n_b = active_mask.sum(dim=(0,1))
        masked_n_ab = masked_n_a + masked_n_b
        masked_sum_b = masked_diffs.sum(dim=(0,1))
        masked_mean_b = (masked_sum_b / masked_n_b).nan_to_num() # TODO: is this cool??
        masked_mean_a = self.masked_means
        masked_delta = masked_mean_b - masked_mean_a
        masked_mean_ab = masked_mean_a + masked_delta * (masked_n_b / masked_n_ab)
        masked_m2_a = self.masked_m2
        masked_m2_b = (masked_mean_b - masked_diffs).pow(2).sum(dim=(0,1))
        masked_m2_ab = (
            masked_m2_a
            + masked_m2_b
            + masked_delta.pow(2) * masked_n_a * masked_n_b / masked_n_ab
        )
        
        # update self vars
        self.n_total = n_ab
        self.mean_diffs = mean_ab
        self.sum_of_squared_f2_diffs += curr_diffs.pow(2).sum(dim=(0,1))
        self.m2_diffs = m2_ab

        self.masked_n = masked_n_ab
        self.masked_means = masked_mean_ab
        self.masked_m2 = masked_m2_ab
        self.sum_of_squared_masked_f2_diffs += masked_diffs.pow(2).sum(dim=(0,1))
        self.sum_unablated_f2 += self.second_layer_unablated_acts.sum(dim=(0,1))
        self.sum_ablated_f2 += self.second_layer_ablated_acts.sum(dim=(0,1))
        
    def _finalize(self):
        self.mse = self.sum_of_squared_f2_diffs / self.n_total
        self.masked_mse = self.sum_of_squared_masked_f2_diffs / self.masked_n
        self.variances = self.m2_diffs / self.n_total
        self.std_devs = t.sqrt(self.variances)
        self.masked_variances = self.masked_m2 / self.masked_n
        self.masked_stdevs = t.sqrt(self.variances)

    def save(self):
        if self.first_layer_idx is None or self.feature_idx is None:
            raise Exception("need to run aggregate() before save()")
        directory = "artefacts/ablations"
        filename_prefix_parts = [
            ('layer', self.first_layer_idx),
            ('feat', self.feature_idx),
            ('num_batches', self.num_batches),
            ('batch_size', self.batch_size)
        ]
        filename_prefix = "__".join(
            ["_".join([attr_name, str(attr_value)]) for attr_name, attr_value in filename_prefix_parts]
        )
        tensor_keys = [key for key, val in self.__dict__.items() if isinstance(val, t.Tensor)]
        print("Saving ablation activation aggs")
        name_to_tensor = {
            name: getattr(self, name)
            for name in tensor_keys
        }
        filename = f"{directory}/{filename_prefix}.pth"
        t.save(name_to_tensor, filename)
            
# %%
def plot_tensor_histogram(tensor, bins=30, cutoff=0.95, tensor_desc="tensor"):
    """
    Plots a histogram of the bottom 99% of elements in a PyTorch tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor.
    bins (int): The number of bins for the histogram.
    """
    # Compute the 99th percentile value
    percentile_value = torch.quantile(tensor, cutoff)
    
    # Filter out the top 1% of values
    filtered_tensor = tensor[tensor <= percentile_value]
    
    # Compute the histogram
    hist = torch.histc(filtered_tensor, bins=bins)
    
    # Calculate bin edges
    min_value = torch.min(filtered_tensor)
    max_value = torch.max(filtered_tensor)
    bin_edges = torch.linspace(min_value, max_value, steps=bins + 1)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1].numpy(), hist.numpy(), width=(max_value - min_value) / bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Bottom {cutoff*100}% of {tensor_desc}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=int, help='first layer idx', required=True)
    parser.add_argument('-f', type=int, help='sae feature idx (within layer)', required=True)
    parser.add_argument('-n', type=int, help='num of batches', default=32)
    parser.add_argument('-b', type=int, help='batch size', default=32)
    parser.add_argument('--dry-run', type=bool, help='dry run (do not save)', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    agg = AblationAggregator(
        first_layer_idx=args.l,
        feature_idx=args.f,
        num_batches=args.n,
        batch_size=args.b,
    )
    agg.aggregate()
    if not args.dry_run:
        agg.save()

        
