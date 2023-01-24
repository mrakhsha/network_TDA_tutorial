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


# %% functions
# the function to calculate connectivity matrix
def conn_cal(epochs, beh, exp_cond, projected):
    mat_all = []
    for cnt_sub in range(len(epochs)):
        # %% load the behavioral data
        dat = fromFile("/Users/mohsen/Documents/GitHub/effort_sensory_analysis/" + beh[cnt_sub])
        block_mode = int(dat.extraInfo["block_mode"])
        if exp_cond == "vis_train":
            if block_mode == 1 or block_mode == 2:
                condition = "first_train"
            elif block_mode == 3 or block_mode == 4:
                condition = "second_train"
        elif exp_cond == "vis_test":
            if block_mode == 1 or block_mode == 2:
                condition = "first_test"
            elif block_mode == 3 or block_mode == 4:
                condition = "second_test"
        elif exp_cond == "stim_train":
            if block_mode == 1 or block_mode == 2:
                condition = "second_train"
            elif block_mode == 3 or block_mode == 4:
                condition = "first_train"
        elif exp_cond == "stim_test":
            if block_mode == 1 or block_mode == 2:
                condition = "second_test"
            elif block_mode == 3 or block_mode == 4:
                condition = "first_test"
        elif exp_cond == "both_train":
            condition = "third_train"

        if projected is True:
            label_ts = np.load("/Users/mohsen/Documents/GitHub/effort_sensory_analysis/" + epochs[cnt_sub] + "src_data_" + condition + "_clean_flip.npy")
        elif projected is False:
            label_ts = np.load("/Users/mohsen/Documents/GitHub/effort_sensory_analysis/" + epochs[cnt_sub] + "src_data_" + condition + "_clean.npy")
            label_ts = np.swapaxes(label_ts, 0, 1)
            # remove the last brain area that corresponds to unknown
            label_ts = np.delete(label_ts, (-1), axis=1)

        if condition == "first_train" or condition == "second_train" or condition == "third_train":
            tmax_var = 8
        else:
            tmax_var = 6

        con = spectral_connectivity_epochs(label_ts,
                                           method=con_methods,
                                           mode='multitaper',
                                           sfreq=sfreq, fmin=fmin, fmax=fmax,
                                           tmin=2, tmax=tmax_var,  # we don't want the first two seconds, because they were baseline
                                           faverage=True,
                                           mt_adaptive=True,
                                           n_jobs=16)

        # con is a 3D array, get the connectivity for the first (and only) freq. band
        # for each method
        con_res = dict()
        con_res = con.get_data(output='dense')[:, :, 0]

        # make the matrix symmetrical
        matrix = con_res + con_res.T - np.diag(np.diag(con_res))
        mat_all.append(matrix)

    mat_all = np.asarray(mat_all)
    avg_mat_all = np.nanmean(mat_all, axis=0)
    return mat_all, avg_mat_all


# the function calculates graph theory mesures
def graph_meas(matrix, plot_figs):
    measures = {}
    # Obtaining name of areas according to matching file
    lineList = [line.rstrip('\n') for line in
                open('./1000_Functional_Connectomes/Region Names/region_names_abbrev_file_MR.txt')]

    # Obtaining a random list of numbers to simulate subnetworks -- THESE NUMBERS DO NOT CORRESPOND TO ANY REAL CLASSIFICATION
    sublist = [line.rstrip('\n') for line in open('./subnet_ordernames_MR.txt')]

    # Obtaining a random list of colors that will match the random subnetwork classification for further graphs -- THESE COLORNAMES DO NOT CORRESPOND TO ANY REAL CLASSIFICATION
    colorlist = [line.rstrip('\n') for line in open('./subnet_order_colors_MR.txt')]

    # Obtaining a random list of colors (in numbers) that will match the random subnetwork classification for further graphs -- THESE NUMBERS DO NOT CORRESPOND TO ANY REAL CLASSIFICATION
    colornumbs = np.genfromtxt('./subnet_colors_number_MR.txt')

    # Creating a DataFrame which will have the rows and column names according to the brain areas
    matrixdiagNaN = matrix.copy()
    np.fill_diagonal(matrixdiagNaN, np.nan)
    Pdmatrix = pd.DataFrame(matrixdiagNaN)
    Pdmatrix.columns = lineList
    Pdmatrix.index = lineList
    Pdmatrix = Pdmatrix.sort_index(0).sort_index(1)

    # This mask variable gives you the possibility to plot only half of the correlation matrix.
    mask = np.zeros_like(Pdmatrix.values, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    if plot_figs:

        plt.figure(figsize=(20, 20))
        _ = sns.heatmap(Pdmatrix, cmap='coolwarm', cbar=True, square=False,
                        mask=None)  # To apply the mask, change to mask=mask
        plt.show()

    # Absolutise for further user
    matrix = abs(matrix)
    matrixdiagNaN = abs(matrixdiagNaN)

    # Weight distribution plot
    bins = np.arange(np.sqrt(len(np.concatenate(matrix))))
    bins = (bins - np.min(bins)) / np.ptp(bins)
    if plot_figs:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Distribution of raw weights
        rawdist = sns.distplot(matrixdiagNaN.flatten(), bins=bins, kde=False, ax=axes[0], norm_hist=True)
        rawdist.set(xlabel='Correlation Values', ylabel='Density Frequency')

        # Probability density of log10

        log10dist = sns.distplot(np.log10(matrixdiagNaN).flatten(), kde=False, ax=axes[1], norm_hist=True)
        log10dist.set(xlabel='log(weights)')
        plt.show()

    # Creating a graph
    G = nx.from_numpy_matrix(matrix)

    # Removing self-loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    ##### Now, we compute the network's density.
    # Create graphs for comparison
    matrix2 = matrix.copy()
    matrix3 = matrix.copy()

    # Create sparser graphs
    matrix2[matrix2 <= 0.15] = 0
    matrix3[matrix3 <= 0.2] = 0

    st50G = nx.from_numpy_matrix(matrix2)
    st25G = nx.from_numpy_matrix(matrix3)

    st50G.remove_edges_from(list(nx.selfloop_edges(st50G)))
    st25G.remove_edges_from(list(nx.selfloop_edges(st25G)))

    # Compute densities
    alltoall = nx.density(G)
    st50 = nx.density(st50G)
    st25 = nx.density(st25G)

    names = ['All-To-All', '> 0.15', '> 0.2']
    values = [alltoall, st50, st25]

    dens_dict = dict(zip(names, values))
    measures["density"] = dens_dict

    ##### Now, we compute the nodal degree/strength.
    # Computation of nodal degree/strength
    # print(nx.degree.__doc__)

    strength = G.degree(weight='weight')
    strengths = {node: val for (node, val) in strength}
    nx.set_node_attributes(G, dict(strength), 'strength')  # Add as nodal attribute

    # Normalized node strength values 1/N-1
    normstrenghts = {node: val * 1 / (len(G.nodes) - 1) for (node, val) in strength}
    nx.set_node_attributes(G, normstrenghts, 'strengthnorm')  # Add as nodal attribute

    # Computing the mean degree of the network
    normstrengthlist = np.array([val * 1 / (len(G.nodes) - 1) for (node, val) in strength])
    mean_degree = np.sum(normstrengthlist) / len(G.nodes)
    print(mean_degree)
    measures["mean_degree"] = mean_degree

    ##### Next, we will compute the centralities!
    # Closeness centrality
    # print(nx.closeness_centrality.__doc__)

    # The function accepts a argument 'distance' that, in correlation-based networks, must be seen as the inverse ...
    # of the weight value. Thus, a high correlation value (e.g., 0.8) means a shorter distance (i.e., 0.2).
    G_distance_dict = {(e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data='weight')}

    # Then add them as attributes to the graph edges
    nx.set_edge_attributes(G, G_distance_dict, 'distance')

    # Computation of Closeness Centrality
    closeness = nx.closeness_centrality(G, distance='distance')

    # Now we add the closeness centrality value as an attribute to the nodes
    nx.set_node_attributes(G, closeness, 'closecent')

    # Visualise  values directly
    # print(closeness)

    # # Closeness Centrality Histogram
    # sns.distplot(list(closeness.values()), kde=False, norm_hist=False)
    # plt.xlabel('Centrality Values')
    # plt.ylabel('Counts')
    # plt.show()

    # Betweenness centrality:
    # print(nx.betweenness_centrality.__doc__)
    betweenness = nx.betweenness_centrality(G, weight='distance', normalized=True)

    # Now we add the it as an attribute to the nodes
    nx.set_node_attributes(G, betweenness, 'bc')

    # Visualise  values directly
    # print(betweenness)

    # # Betweenness centrality Histogram
    # sns.distplot(list(betweenness.values()), kde=False, norm_hist=False)
    # plt.xlabel('Centrality Values')
    # plt.ylabel('Counts')
    # plt.show()

    # Eigenvector centrality
    # print(nx.eigenvector_centrality.__doc__)
    eigen = nx.eigenvector_centrality(G, weight='weight')

    # Now we add the it as an attribute to the nodes
    nx.set_node_attributes(G, eigen, 'eigen')

    # Visualise  values directly
    # print(eigen)

    # # Eigenvector centrality Histogram
    # sns.distplot(list(eigen.values()), kde=False, norm_hist=False)
    # plt.xlabel('Centrality Values')
    # plt.ylabel('Counts')
    # plt.show()

    # Page Rank
    # print(nx.pagerank.__doc__)
    pagerank = nx.pagerank(G, weight='weight')

    # Add as attribute to nodes
    nx.set_node_attributes(G, pagerank, 'pg')

    # Visualise values directly
    # print(pagerank)

    # # Page Rank Histogram
    # sns.distplot(list(pagerank.values()), kde=False, norm_hist=False)
    # plt.xlabel('Pagerank Values')
    # plt.ylabel('Counts')
    # plt.show()

    ##### Now, let's move on to the Path Length!
    # Path Length
    # print(nx.shortest_path_length.__doc__)

    # This is a versatile version of the ones below in which one can define or not source and target. Remove the hashtag to use this version.

    measures["shortest_path_length"] = list(nx.shortest_path_length(G, weight='distance'))
    # # This one can also be used if defining source and target:
    # # print(nx.dijkstra_path_length.__doc__)
    # nx.dijkstra_path_length(G, source=45, target=49, weight='distance')
    #
    # # Whereas this one is for all pairs. Remove the hashtag to use this version.
    # # print(nx.all_pairs_dijkstra_path_length.__doc__)
    # list(nx.all_pairs_dijkstra_path_length(G, weight='distance'))

    # Average Path Length or Characteristic Path Length
    # print(nx.average_shortest_path_length.__doc__)
    measures["average_shortest_path_length"] = nx.average_shortest_path_length(G, weight='distance')

    ##### Now, modularity, assortativity, clustering coefficient and the minimum spanning tree!
    # Modularity
    # print(community.best_partition.__doc__)
    # from community import best_partition
    part = community.best_partition(G, weight='weight')
    nx.set_node_attributes(G, part, 'part')
    # Visualise values directly
    # print(part)

    # Check the number of communities
    measures["num_partitions"] = len(set(part.values()).union())

    # Assortativity
    # print(nx.degree_pearson_correlation_coefficient.__doc__)
    nx.degree_pearson_correlation_coefficient(G, weight='weight')

    # Clustering Coefficient
    # print(nx.clustering.__doc__)
    clustering = nx.clustering(G, weight='weight')

    # Add as attribute to nodes
    nx.set_node_attributes(G, clustering, 'cc')

    # Visualise values directly
    print(clustering)

    # # Clustering Coefficient Histogram
    # sns.distplot(list(clustering.values()), kde=False, norm_hist=False)
    # plt.xlabel('Clustering Coefficient Values')
    # plt.ylabel('Counts')
    # plt.show()

    # Average Clustering Coefficient
    # print(nx.clustering.__doc__)

    measures["average_clustering"] = nx.average_clustering(G, weight='weight')
    # Minimum Spanning Tree
    GMST = nx.minimum_spanning_tree(G, weight='distance')

    # Add brain areas as attribute of nodes
    nx.set_node_attributes(G, Convert(lineList), 'area')

    # Add node colors
    nx.set_node_attributes(G, Convert(colorlist), 'color')

    # Add subnetwork attribute
    nx.set_node_attributes(G, Convert(sublist), 'subnet')

    # Add node color numbers
    nx.set_node_attributes(G, Convert(colornumbs), 'colornumb')

    # # Standard Network graph with nodes in proportion to Graph degrees
    # plt.figure(figsize=(30, 30))
    # edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]
    # pos = nx.spring_layout(G, scale=5)
    # nx.draw(G, pos, with_labels=True, width=np.power(edgewidth, 2), edge_color='grey',
    #         node_size=normstrengthlist * 20000,
    #         labels=Convert(lineList), font_color='black', node_color=colornumbs / 10, cmap=plt.cm.Spectral, alpha=0.7,
    #         font_size=9)
    # plt.show()
    # plt.savefig('network.jpeg')

    # Let's visualise the Minimum Spanning Tree
    if plot_figs:
        plt.figure(figsize=(15, 15))
        nx.draw(GMST, with_labels=True, alpha=0.7, font_size=9)
        plt.show()

    # Visualisation of Communities/Modularity
    if plot_figs:
        plt.figure(figsize=(25, 25))
        values = [part.get(node) for node in G.nodes()]
        clust = [i * 9000 for i in nx.clustering(G, weight='weight').values()]
        nx.draw(G, pos=community_layout(G, part), font_size=8, node_size=clust, node_color=values,
                width=np.power([d['weight'] for (u, v, d) in G.edges(data=True)], 2),
                with_labels=True, labels=Convert(lineList), font_color='black', edge_color='grey', cmap=plt.cm.Spectral,
                alpha=0.7)
        plt.show()
    # save the graph information
    measures["graph"] = G

    return measures


# the function that goes through all subjects to calculate graph measures
def subs_graph_meas(matrix, plot_figs):
    meas_all = []

    if len(matrix.shape) == 2:
        matrix = np.reshape(matrix, (1, matrix.shape[0], matrix.shape[1]))

    for cnt_sub in range(matrix.shape[0]):
        this_matrix = matrix[cnt_sub, :, :]
        measures = graph_meas(this_matrix, plot_figs)
        meas_all.append(measures)

    return meas_all

# Function to transform our list of brain areas into a dictionary
def Convert(lst):
    res_dct = {i : lst[i] for i in range(0, len(lst))}
    return res_dct


# How to get node positions according to
# https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx
def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos
# %% subject list
# list_beh = ["AB04_2022-10-12_16h05.30.004.psydat"]
list_beh = ["AB02_2022-10-07_10h54.53.432.psydat",
            "AB04_2022-10-12_16h05.30.004.psydat",
            "AB05_2022-10-18_14h57.19.640.psydat",
            "AB06_2022-10-20_10h59.44.034.psydat",
            "AB08_2022-10-26_15h52.56.723.psydat",
            "AB10_2022-11-03_11h50.08.761.psydat",
            "AB11_2022-11-11_11h55.06.119.psydat"]
# %% epochs list
# epoch_list = ["AB04_"]
epoch_list = ["AB02_",
              "AB04_",
              "AB05_",
              "AB06_",
              "AB08_",
              "AB10_",
              "AB11_"]
# %% declare variables
sfreq = 1000.0
fmin = 8
fmax = 13
con_methods = ['wpli']
plot_figs_par = False
# %% calculate graph measures
mat_all_vis_train, avg_mat_all_vis_train = conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="vis_train", projected=True)
meas_all_vis_train = subs_graph_meas(mat_all_vis_train, plot_figs=plot_figs_par)

mat_all_vis_test, avg_mat_all_vis_test = conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="vis_test", projected=True)
meas_all_vis_test = subs_graph_meas(mat_all_vis_test, plot_figs=plot_figs_par)

mat_all_stim_train, avg_mat_all_stim_train = conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="stim_train", projected=True)
meas_all_stim_train = subs_graph_meas(mat_all_stim_train, plot_figs=plot_figs_par)

mat_all_stim_test, avg_mat_all_stim_test = conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="stim_test", projected=True)
meas_all_stim_test = subs_graph_meas(mat_all_stim_test, plot_figs=plot_figs_par)

mat_all_both_train, avg_mat_all_both_train = conn_cal(epochs=epoch_list, beh=list_beh, exp_cond="both_train", projected=True)
meas_all_both_train = subs_graph_meas(mat_all_both_train, plot_figs=plot_figs_par)
# %% do statistics
meas_par = meas_all_vis_train
idx_area = [44, 45, 48, 49]
den_all_15 = []
den_all_20 = []
mean_degree_vec = []
avg_shortest_path_vec = []
num_partition_vec = []
avg_clustering_vec = []
avg_short_area_vec = []
strength_vec = []
norm_strength_vec = []
for cnt_sub in range(len(list_beh)):
    den_all_15.append(meas_par[cnt_sub]["density"]["> 0.15"])
    den_all_20.append(meas_par[cnt_sub]["density"]["> 0.2"])
    mean_degree_vec.append(meas_par[cnt_sub]["mean_degree"])
    avg_shortest_path_vec.append(meas_par[cnt_sub]["average_shortest_path_length"])
    num_partition_vec.append(meas_par[cnt_sub]["num_partitions"])
    avg_clustering_vec.append(meas_par[cnt_sub]["average_clustering"])
    b = []
    for this_area in idx_area:
        a = []
        for other_area in idx_area:
            a.append(meas_par[cnt_sub]["shortest_path_length"][this_area][1][other_area])
        b.append(a)
    avg_short_area_vec.append(b)
    st = nx.get_node_attributes(meas_par[cnt_sub]["graph"], "strength")
    norm_st = nx.get_node_attributes(meas_par[cnt_sub]["graph"], "strengthnorm")
    c = []
    d = []
    e = [] # for closeness measure
    for this_area in idx_area:
        c.append(st[this_area])
        d.append(norm_st[this_area])
    strength_vec.append(c)
    norm_strength_vec.append(d)


