#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as t
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
import einops
from dataclasses import dataclass
from typing import List
from functools import partial
import jaxtyping as jt


if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"


# # Clamping SAE features in one layer and measuring effects on the subsequent layer
# ## Plan described [here](https://spar2024.slack.com/archives/C0794GNT8KS/p1719950740186749?thread_ts=1719934219.491869&cid=C0794GNT8KS)

# In[2]:


# loading a small set of correlations to play around with
pearson_0_1_small: 'f' = t.load('../../data/res_jb_sae_feature_correlation_pearson_0_1.pt')


# In[4]:


# find the highest correlations
def create_value_tensor(matrix: 'f,f') -> 'f*f,3':
    m, _ = matrix.shape
    
    # Step 1: Flatten the matrix (shape: [m*m])
    flattened_matrix = matrix.flatten()
    
    # Step 2: Create row and column indices
    row_indices = t.arange(m).repeat_interleave(m)
    col_indices = t.arange(m).repeat(m)
    
    # Step 3: Create the final tensor with indices and values
    values = flattened_matrix
    result = t.stack((row_indices, col_indices, values), dim=1)
    
    # Step 4: Sort the result tensor by values
    sorted_result = result[t.argsort(result[:, 2], descending=True)]
    
    return sorted_result


# In[5]:


ranked_features = create_value_tensor(pearson_0_1_small)


# In[6]:


ranked_features


# In[7]:


# Create a mask to identify rows where the last column is not NaN
not_nan_mask = ~torch.isnan(ranked_features[:, 2])

# Create a mask to identify rows where the last column is less than 1
less_than_one_mask = ranked_features[:, 2] < 0.99

# Combine masks using logical AND
combined_mask = not_nan_mask & less_than_one_mask

# Use the combined mask to filter the rows
filtered_tensor = ranked_features[combined_mask]


# In[8]:


filtered_tensor


# In[9]:


pearson_0_1_small[10715, 20175]


# # Dani:
# probably a way to get the highest values builtin
# 

# ## Measure the correlation when we pass the residual stream that's reconstructed from the SAE features - NO clamping
# As mentioned at the end of June, normally the SAE features values are read by us and discarded - they are not passed back into the model for inference.
# However, if we are going to be clamping an SAE feature and seeing its impact downstream, then we need to first see what happens when we pass the SAE features downstream with no clamping - because the mere act of projecting a residual stream into an SAE space and then back into residual stream space is a lossy operation (even though the SAE is supposed to represent the residual stream)

# In[10]:


model = HookedTransformer.from_pretrained("gpt2-small", device=device)
sae_id_to_sae = {}
for layer in tqdm(list(range(model.cfg.n_layers))):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=device
    )
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    sae_id_to_sae[sae_id] = sae


# In[16]:


f
# These hyperparameters are used to pre-process the data
pre_0_sae_id = "blocks.0.hook_resid_pre"
pre_0_sae = sae_id_to_sae[pre_0_sae_id]
context_size = pre_0_sae.cfg.context_size
prepend_bos = pre_0_sae.cfg.prepend_bos
d_sae = pre_0_sae.cfg.d_sae
batch_size = 32

dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)
token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=context_size,
    add_bos_token=prepend_bos,
)

tokens = token_dataset['tokens']


# In[17]:


# OPTIONAL: Reduce dataset for faster experimentation
num_of_sentences = 1024
tokens = tokens[:num_of_sentences]


# In[19]:


tokens


# In[20]:


tokens.shape


# In[21]:


context_size


# In[13]:


data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)


# In[62]:


# looking at compute-pearson-0.py to figure out how hooks work


# okay so add_hook takes a function...and then...
# https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.hook_points.html#transformer_lens.hook_points.HookPoint.add_hook

# TODO: make a lambda function
# model.reset_hooks()

# TODO: find out where "pre" is defined
# michael: every time you do something with an activation. might not mean anything bc "residual stream" is not an action (unlike, say, adding the attention to the residual)
# https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/full-merm.svg
# model.add_hook("blocks.0.hook_resid_pre", replace_with_sae_output)


# Okay...nice...looks like most of the output logits have changed now $$that this single layer is using SAE activations instead of the residual stream!
# 
# Next step: checking the correlation of 55, 4
# 
# how do I do that...
# okay well how did we compute correlations initially?
# 
# First I need to collect the sae_activations but this time with the SAE being used to influence the second layer's activations

# In[14]:


@dataclass
class LayerFeatures:
    layer_idx: int
    feature_idxes: List[int]

@dataclass
class AggregatorConfig:
    layer_1: LayerFeatures
    layer_2: LayerFeatures
    
class BatchedPearson:
    def __init__(self, agg_conf: AggregatorConfig):
        """Calculates the pair-wise Pearson correlation of two tensors that are provided batch-wise.
        """
        self.agg_conf = agg_conf
        shape = (len(agg_conf.layer_1.feature_idxes), len(agg_conf.layer_2.feature_idxes))
        self.count = 0

        self.sums_1 = torch.zeros(shape[0])
        self.sums_2 = torch.zeros(shape[1])

        self.sums_of_squares_1 = torch.zeros(shape[0])
        self.sums_of_squares_2 = torch.zeros(shape[1])

        self.sums_1_2 = torch.zeros(shape)

        self.nonzero_counts_1 = torch.zeros(shape[0])
        self.nonzero_counts_2 = torch.zeros(shape[1])

    def process(self, tensor_1, tensor_2):
        self.count += tensor_1.shape[-1]

        self.sums_1 += tensor_1.sum(dim=-1)
        self.sums_2 += tensor_2.sum(dim=-1)

        self.sums_of_squares_1 += (tensor_1 ** 2).sum(dim=-1)
        self.sums_of_squares_2 += (tensor_2 ** 2).sum(dim=-1)

        self.sums_1_2 += einops.einsum(tensor_1, tensor_2, 'f1 t, f2 t -> f1 f2')

        self.nonzero_counts_1 += tensor_1.count_nonzero(dim=-1)
        self.nonzero_counts_2 += tensor_2.count_nonzero(dim=-1)

    def finalize(self):
        means_1 = self.sums_1 / self.count
        means_2 = self.sums_2 / self.count

        # Compute the covariance and variances
        covariances = (self.sums_1_2 / self.count) - einops.einsum(means_1, means_2, 'f1, f2 -> f1 f2')

        variances_1 = (self.sums_of_squares_1 / self.count) - (means_1 ** 2)
        variances_2 = (self.sums_of_squares_2 / self.count) - (means_2 ** 2)

        stds_1 = torch.sqrt(variances_1).unsqueeze(1)
        stds_2 = torch.sqrt(variances_2).unsqueeze(0)

        # Compute the Pearson correlation coefficient
        correlations = covariances / stds_1 / stds_2

        return correlations


# In[47]:


agg_confs = [
    AggregatorConfig(
        LayerFeatures(0, [10715]),
        LayerFeatures(1, [20175]),
    ),
]
aggs = [BatchedPearson(agg_conf) for agg_conf in agg_confs]
sae_activations = torch.empty(model.cfg.n_layers, d_sae, batch_size * context_size)

def replace_acts_with_sae(activations: t.Tensor, hook: HookPoint):
    # replaces the residual stream activations with SAE activations
    sae = sae_id_to_sae[hook.name]
    return sae(activations)

def emit_sae(activations: t.Tensor, hook: HookPoint):
    # emits what the SAE activations would be for these input activations
    sae: SAE = sae_id_to_sae[hook.name]
    sae_acts = sae.encode(activations)
    sae_activations[hook.layer()] = einops.rearrange(
        sae_acts,
        'batch seq features -> features (batch seq)'
    )
    return activations

with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        # okay so if we have a hook which is doing SAE embeddings and returning them then...we need to be...
        # i think it'll be easier if BOTH functions save SAEs, but only one ofthem returns the SAEs and other returns original acts
        model.reset_hooks()
        # TODO(optimization): collapse these into a single function which decides whether to 
        # emit the SAE activations or return the originals
        model.add_hook(
            lambda name: name.endswith('.hook_resid_pre'),
            emit_sae,
        )
        model.run_with_hooks(batch_tokens)
        for agg in aggs:
            agg.process(
                sae_activations[agg.agg_conf.layer_1.layer_idx, agg.agg_conf.layer_1.feature_idxes],
                sae_activations[agg.agg_conf.layer_2.layer_idx, agg.agg_conf.layer_2.feature_idxes]
            )
    pearson_correlations = [aggregator.finalize() for aggregator in aggs]


# In[50]:


pearson_correlations[0]


# # That's what I suspected
# Now I'm going to recompute it but I'll see what happens when I feed the lossy first layer into the second layer

# In[53]:


agg_confs = [
    AggregatorConfig(
        LayerFeatures(0, [10715]),
        LayerFeatures(1, [20175]),
    ),
]
aggs = [BatchedPearson(agg_conf) for agg_conf in agg_confs]
sae_activations = torch.empty(model.cfg.n_layers, d_sae, batch_size * context_size)

def replace_acts_with_lossy_sae(activations: t.Tensor, hook: HookPoint):
    # replaces the residual stream activations with SAE activations
    sae = sae_id_to_sae[hook.name]
    return sae(activations)

def emit_sae(activations: t.Tensor, hook: HookPoint):
    # emits what the SAE activations would be for these input activations
    sae: SAE = sae_id_to_sae[hook.name]
    sae_acts = sae.encode(activations)
    sae_activations[hook.layer()] = einops.rearrange(
        sae_acts,
        'batch seq features -> features (batch seq)'
    )
    return activations

with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        # okay so if we have a hook which is doing SAE embeddings and returning them then...we need to be...
        # i think it'll be easier if BOTH functions save SAEs, but only one ofthem returns the SAEs and other returns original acts
        model.reset_hooks()
        # TODO(optimization): collapse these into a single function which decides whether to 
        # emit the SAE activations or return the originals
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    lambda name: name.endswith('.hook_resid_pre'), # we emit the SAE activations no matter what
                    emit_sae,
                ),
                (
                    "blocks.0.hook_resid_pre", # TODO: is this right?? is this just a token + positional embedding??
                    replace_acts_with_lossy_sae,
                )
            ]
        )
        for agg in aggs:
            agg.process(
                sae_activations[agg.agg_conf.layer_1.layer_idx, agg.agg_conf.layer_1.feature_idxes],
                sae_activations[agg.agg_conf.layer_2.layer_idx, agg.agg_conf.layer_2.feature_idxes]
            )
    pearson_correlations = [aggregator.finalize() for aggregator in aggs]


# In[54]:


pearson_correlations[0]


# Hmmm...they're even *more* correlated.
# Is that what I would expect? Maybe...if now, the layer1 feature value is lower on avg, so it would be more "in line" with layer 0?

# ## Extending this to work on multiple feature pairs within a single layer pair
# All I'd need to do is extend AggConf to operate on multiple pairs of features

# ## Extending this to work across multiple layer pairs
# 
# the naive way to do this would be something like:

# In[ ]:


def emit_sae_from_lossy_reconstruction(activations, hook, layer_idx_being_reconstructed):
    sae_activations[layer_idx_being_reconstructed][hook.layer()] = einops.rearrange(
        sae_acts,
        'batch seq features -> features (batch seq)'
    )

sae_activations = t.empty(
    model.cfg.n_layers, # represents which layer is being reconstructed in this pass
    model.cfg.n_layers, # represents which layer's activations are being collected in these values
    d_sae, # the index of each activation
    batch_size * context_size # the tokens which are contributing to the activations
)
for layer_idx_to_lossily_reconstruct in range(len(layers)):
    model.run_with_hooks(
        (
            lambda name: name.endswith('.hook_resid_pre'),
            partial(emit_sae, layer_idx_to_lossily_reconstruct),
        ),
        (
            f"blocks.{layer_idx_to_lossily_reconstruct}.hook_resid_pre",
            replace_acts_with_lossy_sae,
        )
    )


# # Measuring ablation impact on correlations
# 
# Okay, so now we're able to reconstruct residual streams out of SAEs without having done anything to do the SAEs. But what we really want to do is to ablate one of the SAE features and see what that does to the correlations.

# In[39]:


# unablated_f2_acts = t.empty(num_tokens)
# ablated_f2_acts = t.empty(num_tokens)


"""
L0F1-> L1F2
1. what is F2's value when F1 is not ablated
2. what is F2's value when F1 is ablated
x = input/tokens
m(x) = f2(x) - f2(x|a)
m2 = mse(m(x) for x in all_inputs) 


1. first layer, we save the error terms
2. in the second layer, we store the desired feature's activations

in the second pass
1. in the first layer, we ablate, reconstruct, and add the error term, return
2. in the second layer, we store feature activations after receiving ablated input
"""
sae_errors = t.empty(batch_size, context_size, model.cfg.d_model)
feature_2_unablated_acts = t.empty(batch_size, context_size)
feature_2_ablated_acts = t.empty(batch_size, context_size)

# TODO: rename
def save_error_terms(activations: t.Tensor, hook: HookPoint):
    global sae_errors
    sae: SAE = sae_id_to_sae[hook.name]
    reconstructed_acts = sae(activations)
    sae_errors = activations - reconstructed_acts 
    return activations
    
def save_feature_2_acts(activations: t.Tensor, hook: HookPoint, feature_idx: int, ablated: bool):
    global feature_2_ablated_acts
    global feature_2_unablated_acts
    sae: SAE = sae_id_to_sae[hook.name]
    sae_feats = sae.encode(activations)
    if ablated:
        feature_2_ablated_acts = sae_feats[:,:,feature_idx]
    else:
        feature_2_unablated_acts = sae_feats[:,:,feature_idx]
    return activations

def ablate_and_reconstruct_with_errors(activations: t.Tensor, hook: HookPoint, feature_idx: int):
    global sae_errors
    sae: SAE = sae_id_to_sae[hook.name]
    sae_feats = sae.encode(activations)
    sae_feats[:,:,feature_idx] = 0    
    return sae.decode(sae_feats) + sae_errors # we assume that the hooks are called in the correct order s.t. these errors correspond to the same tokens

with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.reset_hooks()
        # collect the unablated activations
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    "blocks.0.hook_resid_pre", # TODO: parameterize
                    save_error_terms,
                ),
                (
                    "blocks.1.hook_resid_pre", # TODO: parameterize
                    partial(save_feature_2_acts, feature_idx=20175, ablated=False) # TODO :parameterize
                )
            ]
        )
        # collect the activations after ablation
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    "blocks.0.hook_resid_pre", # TODO: parameterize
                    partial(ablate_and_reconstruct_with_errors, feature_idx=10715)
                ),
                (
                    "blocks.1.hook_resid_pre", # TODO: parameterize
                    partial(save_feature_2_acts, feature_idx=20175, ablated=True) # TODO :parameterize
                )
            ]
        )
        break
# # adjacent_feat_pairs 
# def compute_ablation_impact(pairs: [(layer_0, feat_51, feat_853)], tokens):
#     """
#     1. take layer 1 resid
#         a. compute reconstruction error
#             error = resid - sae(resid)
#         b. ablate a SAE feature
#             sae_acts = sae.encode(resid)
#             sae_acts[sae_feat_idx] = 0
#     2. return sae.decode(sae_acts) + error
#     """
#     return [
#         [(layer_0, feat_51, feat_853, )]
#     ]


# In[46]:


diff = (feature_2_ablated_acts - feature_2_unablated_acts)


# In[ ]:





# In[54]:


feature_2_unablated_acts.count_nonzero()


# In[55]:


feature_2_ablated_acts.count_nonzero()


# In[51]:


torch.set_printoptions(threshold=4000)
diff.count_nonzero()


# In[ ]:





# ## NEXT STEPS:
# 1. figure out how to accumulate over batches
# 2. 

# In[ ]:


# let's say that we've come up with a bunch of (first_layer_idx, first_layer_feature_idx, next_layer_feature_idx, correlation) tuples e.g.
correlations = [
    (0, 10305, 403, 0.95), # feature 10305 in layer 0 and feature 403 in layer 1 have a 0.95 correlation
]
# and now we want to see "what would happen if we ablated the first features to 0? how would that affect each second feature? would they still be highly correlated?"
# so we'd want to end up with something like
ablated_correlations = [
    (0, 10305, 403, 0.90) # hm.actually...what WOULD we expect this to be?
]


# ## TODO: what do we expect the new correlations to be between previously-correlated feature pairs after we ablate the first feature in the pair? if it used to look like
# ```
# feature_1_feature_2_acts = [
#     (4, 4.2),
#     (3.1, 2.9),
#     (5.5, 5.7),
#     (0.5, 0.8)
# ]
# ```
# and now it's like
# ```
# feature_1_feature_2_acts = [
#     (0, 0.5)
#     (0, 0.8),
#     (0, 0.5),
#     (0, 0.5)
# ]
# ```
# 
# then that indicates that feature 1 DOES cause feature 2, right? but wouldn't these have a correlation of nan? so what am I supposed to measure?

# https://spar2024.slack.com/archives/C078944NFD4/p1720459095680359
