# %%
import os
import numpy as np
import sys
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from matplotlib.ticker import LinearLocator

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import get_filename, load_similarity_data


# %%
# Load similarity matrices
measure_name = "pearson_correlation"
sae_name = 'res_jb_sae'
n_layers = 12
activation_threshold = None
n_tokens = '10M'

folder = f'../../artefacts/similarity_measures/{measure_name}/.unclamped'

similarities = load_similarity_data([f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens=n_tokens, first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)])


# %%
thresholds = np.linspace(0, 1, 21)
high_sim_features = np.arange(15)


# %%
data = []
for threshold in tqdm(thresholds):
    pass_through_connections = (similarities >= threshold)
    n_forward_pass_through = pass_through_connections.sum(axis=-1).flatten()

    data.append([(n_forward_pass_through >= n).sum() for n in high_sim_features] + [(n_forward_pass_through > max(high_sim_features)).sum()])

# %%
data = np.array(data)

n_thresholds, n_high_sim_features = data.shape
    
xpos, ypos = np.meshgrid(range(n_thresholds), range(n_high_sim_features), indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

dx = dy = 1.0
dz = data.ravel()


# %%
# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the bars
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

# Set labels and title
ax.set_xlabel('Number of high-similarity downstream neighbors')
ax.set_ylabel('High-similarity threshold')
ax.set_zlabel('Number of features')
ax.set_zscale('log')
ax.set_title("3D Bar Plot")

# Set tick labels
ax.set_xticks(range(n_high_sim_features))
ax.set_xticklabels([str(n) for n in high_sim_features] + [f'> {max(high_sim_features)}'])
ax.set_yticks(range(n_thresholds))
ax.set_yticklabels([f'{t:.2f}' for t in thresholds])
ax.tick_params(axis='y', rotation=45)

# Adjust the viewing angle for better visibility
ax.view_init(elev=20, azim=45)

fig.show()


# %%
pass_through_connections = (similarities >= 0.9)
n_forward_pass_through = pass_through_connections.sum(axis=-1).flatten()

plt.plot([(n_forward_pass_through >= n).sum() for n in high_sim_features])
plt.yscale('log')


# %%
data[:, -1].shape


# %%
# set up the figure and Axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# fake data
_x = np.arange(4)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = x + y
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Shaded')

plt.show()

# %%
top

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 12))

# Make data.
X = np.arange(data.shape[1])
Y = thresholds
X, Y = np.meshgrid(X, Y)
Z = data

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)


ax.set_xlabel('Number of high-similarity downstream neighbors')
ax.set_ylabel('High-similarity threshold')
ax.set_zlabel('Number of features')
ax.set_zscale('log')
ax.set_title("High-similarity downstream neighbors for Pearson correlation")

ax.set_xticks(range(n_high_sim_features))
ax.set_xticklabels([str(n) for n in high_sim_features] + [f'> {max(high_sim_features)}'])
# ax.set_yticks(range(n_thresholds))
# ax.set_yticklabels([f'{t:.2f}' for t in thresholds])
# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(20, 70, 0)

plt.show()
# %%
