import pickle
import function_network as fn


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