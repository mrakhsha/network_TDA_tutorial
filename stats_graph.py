import pickle
import function_network as fn
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# sns.set(style="whitegrid")
import numpy as np
import pandas as pd

# %% subject list
# list_beh = ["AB04_2022-10-12_16h05.30.004.psydat"]
list_beh = ["AB02_2022-10-07_10h54.53.432.psydat",
            "AB04_2022-10-12_16h05.30.004.psydat",
            "AB05_2022-10-18_14h57.19.640.psydat",
            "AB06_2022-10-20_10h59.44.034.psydat",
            "AB08_2022-10-26_15h52.56.723.psydat",
            "AB10_2022-11-03_11h50.08.761.psydat",
            "AB11_2022-11-11_11h55.06.119.psydat"]


vis_train = open('small_world_vis_train.pickle', 'rb')
meas_all_vis_train = pickle.load(vis_train)

vis_test = open('small_world_vis_test.pickle', 'rb')
meas_all_vis_test = pickle.load(vis_test)

stim_train = open('small_world_stim_train.pickle', 'rb')
meas_all_stim_train = pickle.load(stim_train)

stim_test = open('small_world_stim_test.pickle', 'rb')
meas_all_stim_test = pickle.load(stim_test)

both_train = open('small_world_both_train.pickle', 'rb')
meas_all_both_train = pickle.load(both_train)

# %% summarize results
net_vis_train = fn.network_measures_area(meas_all_vis_train, list_beh=list_beh)
net_vis_test = fn.network_measures_area(meas_all_vis_test, list_beh=list_beh)
net_stim_train = fn.network_measures_area(meas_all_stim_train, list_beh=list_beh)
net_stim_test = fn.network_measures_area(meas_all_stim_test, list_beh=list_beh)
net_both_train = fn.network_measures_area(meas_all_both_train, list_beh=list_beh)

# %% do statistics
data_vis_train = net_vis_train
data_vis_test = net_vis_test
data_stim_train = net_stim_train
data_stim_test = net_stim_test
data_both_train = net_both_train

all_data = [data_vis_train, data_vis_test, data_stim_train, data_stim_test, data_both_train]

all_stats = fn.stats_net_measures(all_data)
# %% create graph measures dataframe across different training conditions
tips = sns.load_dataset("tips")
num_sub = 7
density = np.squeeze(np.concatenate([data_vis_train["den_all_15"], data_stim_train["den_all_15"], data_both_train["den_all_15"]], axis=0))
cluster = np.squeeze(np.concatenate([data_vis_train["avg_clustering_vec"], data_stim_train["avg_clustering_vec"], data_both_train["avg_clustering_vec"]], axis=0))
avg_path = np.squeeze(np.concatenate([data_vis_train["avg_shortest_path_vec"], data_stim_train["avg_shortest_path_vec"], data_both_train["avg_shortest_path_vec"]], axis=0))
condition = np.concatenate([np.full((num_sub,), "visual"), np.full((num_sub,), "vibrotactile"), np.full((num_sub,), "both")])
ID = np.tile(np.arange(0, num_sub, 1, dtype=int), 3)
data_meas = {'density': density, 'cluster': cluster, 'avg_path': avg_path, 'condition': condition, 'ID': ID}
df_meas = pd.DataFrame(data_meas)

# %% plot the density measures across different training conditions
fig, ax = plt.subplots()
clrs = ['blue', 'green', 'magenta']
sns.barplot(x="condition", y="density", data=df_meas, capsize=.1, ci="sd", palette=clrs)
sns.swarmplot(x="condition", y="density", data=df_meas, color="0", alpha=.95)
sns.lineplot(x="condition", y="density", hue='ID', data=df_meas,
             estimator=None, legend=False, palette=['black'])
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
# ax.xaxis.set_tick_params(width=3, length=8)
# ax.yaxis.set_tick_params(width=3, length=8)
ax.set_xlabel("condition", fontsize=18)
ax.set_ylabel("density", fontsize=18)
# change all spines
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks(np.arange(0, 0.9, step=0.2))

plt.ylim([0, 0.8])
plt.grid(False)
fig.savefig("Figures/density.tiff", format='tiff', bbox_inches='tight', pad_inches=0.1)
plt.show()
# %% plot the clustering measures across different training conditions
fig, ax = plt.subplots()
clrs = ['blue', 'green', 'magenta']
sns.barplot(x="condition", y="cluster", data=df_meas, capsize=.1, ci="sd", palette=clrs)
sns.swarmplot(x="condition", y="cluster", data=df_meas, color="0", alpha=.95)
sns.lineplot(x="condition", y="cluster", hue='ID', data=df_meas,
             estimator=None, legend=False, palette=['black'])
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
# ax.xaxis.set_tick_params(width=3, length=8)
# ax.yaxis.set_tick_params(width=3, length=8)
ax.set_xlabel("condition", fontsize=18)
ax.set_ylabel("clustering coefficient", fontsize=18)
# change all spines
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks(np.arange(0, 0.5, step=0.1))

plt.ylim([0, 0.5])
plt.grid(False)
fig.savefig("Figures/clsutering.tiff", format='tiff', bbox_inches='tight', pad_inches=0.1)
plt.show()
# %% plot the average path across different training conditions
fig, ax = plt.subplots()
clrs = ['blue', 'green', 'magenta']
sns.barplot(x="condition", y="avg_path", data=df_meas, capsize=.1, ci="sd", palette=clrs)
sns.swarmplot(x="condition", y="avg_path", data=df_meas, color="0", alpha=.95)
sns.lineplot(x="condition", y="avg_path", hue='ID', data=df_meas,
             estimator=None, legend=False, palette=['black'])
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
# ax.xaxis.set_tick_params(width=3, length=8)
# ax.yaxis.set_tick_params(width=3, length=8)
ax.set_xlabel("condition", fontsize=18)
ax.set_ylabel("average path length", fontsize=18)
# change all spines
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks(np.arange(0, 12, step=2))

plt.ylim([0, 12])
plt.grid(False)
fig.savefig("Figures/avg_path.tiff", format='tiff', bbox_inches='tight', pad_inches=0.1)
plt.show()
# %% heatmap of the shortest path across the six regions

fig, axn = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18,6))
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for i, ax in enumerate(axn.flat):
    if i == 0:
        df_plt = data_vis_train["mean_avg_short_area_vec"]

    elif i == 1:
        df_plt = data_stim_train["mean_avg_short_area_vec"]

    elif i == 2:
        df_plt = data_both_train["mean_avg_short_area_vec"]

    sns.heatmap(df_plt, ax=ax,
                cbar=i == 0,
                vmin=0, vmax=10,
                cbar_ax=None if i else cbar_ax, annot=True)

fig.tight_layout(rect=[0, 0, .9, 1])
fig.savefig("Figures/avg_path_6reg.tiff", format='tiff', bbox_inches='tight', pad_inches=0.1)
plt.show()

