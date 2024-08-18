# %%
import torch as t
# %%

# %%
b = t.tensor([0.000123456789, 0.123456789, 1234.56789, 0.00000123456789, 123456.7890, 65519, 65520], dtype=t.float16)
for b_elem in b:
    print(f'{b_elem.item():.10f}')
# %%
base = 1.23456789
exponents = t.arange(-9, 6)
floats = base * t.float_power(10, exponents)
floats_16 = t.tensor(floats, dtype=t.float16)
diffs = floats - floats_16
ratios = diffs / floats
print('exponent    float32    float16    diff    ratio')
for i in range(len(exponents)):
    print(f'{exponents[i]}    {floats[i]}    {floats_16[i]}    {diffs[i]}    {ratios[i]}')
# %%
def get_memory_usage(tensor):
    return tensor.element_size() * tensor.numel()
# %%
float32_results = t.zeros((11, 24576, 24576))
# %%
a = 0.00000123456789 * t.ones((10000000, 11), dtype=t.float32)
b = 0.00000123456789 * t.ones((10000000, 11), dtype=t.float16)

a_mean = a.mean()
b_mean = b.mean()
print(f'{a_mean.item():.10f}')
print(f'{b_mean.item():.10f}')
print(f'Error: {100*(b_mean - a_mean)/a_mean:.10f}%')

c = 0.00000123456789 * t.ones((1), dtype=t.float32)
d = 0.00000123456789 * t.ones((1), dtype=t.float16)
error = (d - c)/c
print(f'Error: {100*error.item():.10f}%')
# %%
# %%