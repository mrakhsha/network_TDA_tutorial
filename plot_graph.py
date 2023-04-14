import numpy as np
import mne
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import networkx as nx
from nxviz import CircosPlot
import community
import gudhi
from psychopy.misc import fromFile
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import pickle
import function_network as fn

# %% subject list
list_beh = ["AB04_2022-10-12_16h05.30.004.psydat"]
# list_beh = ["AB02_2022-10-07_10h54.53.432.psydat",
#             "AB04_2022-10-12_16h05.30.004.psydat",
#             "AB05_2022-10-18_14h57.19.640.psydat",
#             "AB06_2022-10-20_10h59.44.034.psydat",
#             "AB08_2022-10-26_15h52.56.723.psydat",
#             "AB10_2022-11-03_11h50.08.761.psydat",
#             "AB11_2022-11-11_11h55.06.119.psydat"]
# %% epochs list
epoch_list = ["AB04_"]
# epoch_list = ["AB02_",
#               "AB04_",
#               "AB05_",
#               "AB06_",
#               "AB08_",
#               "AB10_",
#               "AB11_"]
# %% declare variables
sfreq = 1000.0
fmin = 8
fmax = 13
con_methods = ['wpli']
plot_figs_par = False
# %% calculate graph measures
mat_all_vis_train, avg_mat_all_vis_train = fn.conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="vis_train",
                                                       projected=True, con_methods=con_methods, sfreq=sfreq, fmin=fmin, fmax=fmax)
graph_all_vis_train = fn.make_graph(avg_mat_all_vis_train)

fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# Create a gridspec for adding subplots of different sizes
axgrid = fig.add_gridspec(5, 4)

ax0 = fig.add_subplot(axgrid[0:3, :])
Gcc = graph_all_vis_train.subgraph(sorted(nx.connected_components(graph_all_vis_train), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc, seed=10396953)
nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
ax0.set_title("Connected components of G")
ax0.set_axis_off()
plt.show()

deg_weight = nx.degree(graph_all_vis_train, weight="weight")

deg_weight[1]