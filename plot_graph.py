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
from network_analyses_main import conn_cal
# %% functions
# the function calculates graph theory mesures
def make_graph(matrix):
    # Absolutise for further user
    matrix = abs(matrix)

    # Creating a graph
    G = nx.from_numpy_matrix(matrix)

    # Removing self-loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    return G

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
mat_all_vis_train, avg_mat_all_vis_train = conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="vis_train", projected=True)
graph_all_vis_train = make_graph(mat_all_vis_train)

