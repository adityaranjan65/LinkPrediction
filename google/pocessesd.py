import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle


gplus_EGO_USERS = [100518419853963396365]

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