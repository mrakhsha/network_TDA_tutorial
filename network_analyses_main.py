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
plot_figs_par = True
# %% calculate graph measures
mat_all_vis_train, avg_mat_all_vis_train = fn.conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="vis_train",
                                                       projected=True, con_methods=con_methods, sfreq=sfreq, fmin=fmin, fmax=fmax)
meas_all_vis_train = fn.subs_graph_meas(mat_all_vis_train, plot_figs=plot_figs_par)

mat_all_vis_test, avg_mat_all_vis_test = fn.conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="vis_test",
                                                     projected=True, con_methods=con_methods, sfreq=sfreq, fmin=fmin, fmax=fmax)
meas_all_vis_test = fn.subs_graph_meas(mat_all_vis_test, plot_figs=plot_figs_par)

mat_all_stim_train, avg_mat_all_stim_train = fn.conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="stim_train",
                                                         projected=True, con_methods=con_methods, sfreq=sfreq, fmin=fmin, fmax=fmax)
meas_all_stim_train = fn.subs_graph_meas(mat_all_stim_train, plot_figs=plot_figs_par)

mat_all_stim_test, avg_mat_all_stim_test = fn.conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="stim_test",
                                                       projected=True, con_methods=con_methods, sfreq=sfreq, fmin=fmin, fmax=fmax)
meas_all_stim_test = fn.subs_graph_meas(mat_all_stim_test, plot_figs=plot_figs_par)

mat_all_both_train, avg_mat_all_both_train = fn.conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="both_train",
                                                         projected=True, con_methods=con_methods, sfreq=sfreq, fmin=fmin, fmax=fmax)
meas_all_both_train = fn.subs_graph_meas(mat_all_both_train, plot_figs=plot_figs_par)
# %% summarize results
# net_vis_train = fn.network_measures_area(meas_par=meas_all_vis_train, list_beh=list_beh)
# net_vis_test = fn.network_measures_area(meas_par=meas_all_vis_test, list_beh=list_beh)
# net_stim_train = fn.network_measures_area(meas_par=meas_all_stim_train, list_beh=list_beh)
# net_stim_test = fn.network_measures_area(meas_par=meas_all_stim_test, list_beh=list_beh)
# net_both_train = fn.network_measures_area(meas_par=meas_all_both_train, list_beh=list_beh)
# # %% do statistics
# data_vis_train = net_vis_train
# data_vis_test = net_vis_test
# data_stim_train = net_stim_train
# data_stim_test = net_stim_test
# data_both_train = net_both_train
#
# all_data = [data_vis_train, data_vis_test, data_stim_train, data_stim_test, data_both_train]
#
# all_stats = fn.stats_net_measures(all_data)

with open('small_world_vis_train.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(meas_all_vis_train, f, pickle.HIGHEST_PROTOCOL)

with open('small_world_vis_test.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(meas_all_vis_test, f, pickle.HIGHEST_PROTOCOL)

with open('small_world_stim_train.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(meas_all_stim_train, f, pickle.HIGHEST_PROTOCOL)

with open('small_world_stim_test.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(meas_all_stim_test, f, pickle.HIGHEST_PROTOCOL)

with open('small_world_both_train.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(meas_all_both_train, f, pickle.HIGHEST_PROTOCOL)