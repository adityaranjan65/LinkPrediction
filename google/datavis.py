import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle
from math import log
import matplotlib.pyplot as plt
#from IPython import get_ipython

#get_ipython().run_line_magic('matplotlib','qt')
gplus_EGO_USERS = [100466178325794757407]
def save_visualization(g, file_name, title):
    plt.figure(figsize=(18,18))
    degrees = nx.degree(g)
    #print(list(dict(degrees).keys()))
    # Draw networkx graph -- scale node size by log(degree+1)
    
    nx.draw_spring(g, with_labels=False, 
                   linewidths=2.0,
                   nodelist=list(dict(degrees).keys()),
                   node_size=[log(degree_val+1) * 100 for degree_val in list(dict(degrees).values())], node_color='r')
            
                   
    
    
    
        
    
    # Create black border around node shapes
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    
#     plt.title(title)
    plt.savefig(file_name)
    plt.clf()
def get_network_statistics(g):
    num_connected_components = nx.number_connected_components(g)
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    density = nx.density(g)
    avg_clustering_coef = nx.average_clustering(g)
    avg_degree = sum(list(dict(g.degree()).values())) / float(num_nodes)
    transitivity = nx.transitivity(g)
    
    if num_connected_components == 1:
        diameter = nx.diameter(g)
    else:
        diameter = None # infinite path length between connected components
    
    network_statistics = {
        'num_connected_components':num_connected_components,
        'num_nodes':num_nodes,
        'num_edges':num_edges,
        'density':density,
        'diameter':diameter,
        'avg_clustering_coef':avg_clustering_coef,
        'avg_degree':avg_degree,
        'transitivity':transitivity
    }
    
    return network_statistics
def save_network_statistics(g, file_name):
    network_statistics = get_network_statistics(g)
    with open(file_name, 'wb') as f:
        pickle.dump(network_statistics, f)
    with open(file_name, 'rb') as f:
        x = pickle.load(f)
    print(x)
# Store all ego graphs in pickle files as (adj, features) tuples
for ego_user in gplus_EGO_USERS:
    edges_dir = './gplus/' + str(ego_user) + '.edges'
    feats_dir = './gplus/' + str(ego_user) + '.allfeat'
  
    # Read edge-list
    f = open(edges_dir,'rb')
    g = nx.read_edgelist(f, nodetype=int)
   

    # Add ego user (directly connected to all other nodes)
    
    g.add_node(ego_user)
    
    f=open('file1.txt','w')
    f.write(str(g.node))
    f.close()
    for node in g.nodes():
        if node != ego_user:
            g.add_edge(ego_user, node)

    # read features into dataframe
    df = pd.read_table(feats_dir, sep=' ', header=None, index_col=0)
    
    # Add features from dataframe to networkx nodes
    
    #print(len(df))
    
    for node_index, features_series in df.iterrows():
        # Haven't yet seen node (not in edgelist) --> add it now
        #print(node_index)
        
        if not g.has_node(int(node_index)):
            
            g.add_node(int(node_index))
            
            g.add_edge(int(node_index),int( ego_user))
            
            
        g.node[int(node_index)]['features'] = features_series.as_matrix()
   
    #print(g.node)
    f=open('degree.txt','w')
    f.write(str(g.node))
    f.close()
    print(len(g.node))
    
  
    assert nx.is_connected(g)
    
    # Get adjacency matrix in sparse format (sorted by g.nodes())
    #print(len(g.node))
    adj = nx.adjacency_matrix(g) 
    

    # Get features matrix (also sorted by g.nodes())
    print(df.shape[0])
    print(df.shape[1])
    features = np.zeros((df.shape[0], df.shape[1]))
    # num nodes, num features
    for i, node in enumerate(g.nodes()):
      
        features[i,:] = g.node[node]['features']
    
    # Save adj, features in pickle file
    print(len(g.node))
    
    
    network_tuple = (adj, features)
    f=open('file.txt','w')
    f.write(str(df.iterrows))
    f.close() 
    with open("g-processed/{0}-adj-feat.pkl".format(ego_user), "wb") as f:
        pickle.dump(network_tuple, f)
    visualization_file_name = './visualizations/g-ego-{0}-visualization.jpeg'.format(ego_user)
    statistics_file_name = './network-statistics/g-ego-{0}-statistics.pdf'.format(ego_user)
    title = 'Facebook Ego Network: ' + str(ego_user)
    
    save_visualization(g, visualization_file_name, title)
    save_network_statistics(g, statistics_file_name)