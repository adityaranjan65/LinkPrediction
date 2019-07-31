import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle


EGO_USER = 100329698645326486178


network_dir = './g-processed/{0}-adj-feat.pkl'.format(EGO_USER)
with open(network_dir, 'rb') as f:
    adj, features = pickle.load(f,encoding='iso-8859-1')
G = nx.Graph(adj)



gplus_EGO_USERS = [100329698645326486178]

# Store all ego graphs in pickle files as (adj, features) tuples
for ego_user in gplus_EGO_USERS:
    edges_dir = './gplus/' + str(ego_user) + '.edges'
    feats_dir = './gplus/' + str(ego_user) + '.allfeat'
  
    # Read edge-list
    f = open(edges_dir,'rb')
    g = nx.read_edgelist(f, nodetype=int)
   

    # Add ego user (directly connected to all other nodes)
    
    g.add_node(ego_user)
    
   
    for node in g.nodes():
        if node != ego_user:
            g.add_edge(ego_user, node)

    # read features into dataframe
    df = pd.read_table(feats_dir, sep=' ', header=None, index_col=0)
    
    
    
    for node_index, features_series in df.iterrows():
        # Haven't yet seen node (not in edgelist) --> add it now
        #print(node_index)
        
        if not g.has_node(int(node_index)):
            
            g.add_node(int(node_index))
            
            g.add_edge(int(node_index),int( ego_user))
            
            
        g.node[int(node_index)]['features'] = features_series.as_matrix()

 
    assert nx.is_connected(g)
    
    # Get adjacency matrix in sparse format (sorted by g.nodes())
    #print(len(g.node))
    adj = nx.adjacency_matrix(g) 
    #y=(2*len(g.edge))/len(g.node)
    #print(y)
    z=0
    l=[]
    # Get features matrix (also sorted by g.nodes())
   
    for node in g.nodes():
        x=g.degree(node)
        l.append(x)
        
        z=x+z
        f=open('degree.txt','a')
    
        f.write(str(node))
        f.write(str(x)+"\n")
        f.close()
    y=z/len(g.node)
    ##print(y)
    print(z)
    #l=list(set(l))
    l.sort()  
    from scipy.stats import poisson
 
#l=list(set(l))
cnt=0
x=[]
print(l)

x.append(33)
for n in range(len(l)):
    
    if(n>=1):
        if not(l[n]==l[n-1]):
            count=l.count(l[n]);
            x.append(count);
j=[]
for n in x:
    n=(n/2247)*100
    
    j.append(math.log(n))
print(j)
l=list(set(l))

sum=0
for n in l:
    sum=sum+pow(n,-2)
print(1/sum)
c=1/sum

z=[]
for n in l:
    val=len(g.node)*0.6*pow(n,-1.8)
    z.append(math.log(val))
#print(z)
a=[]
for n in l:
    
    n=math.log(n)
    a.append(n)
#plt.figure(figsize=(18,18))
plt.ylim((-4,1))
plt.ylabel('Log of degree occurence')
plt.xlabel('Log of degree')
plt.title('LOG LOG PLOT')
#plt.plot(a,z,color="red")

s = 0
for i in x:
    s=s+i
cdf = np.cumsum(x)/s
print(cdf)
ccdf=1-cdf
print(ccdf)
b=[]
for i in ccdf:
    if(i!=0):
        x.append(math.log(i))
plt.plot(range(len(x)),x,'bo')
#plt.scatter(a,j)


'''plt.ylabel('log(distribution)')
plt.xlabel('log(degree)')
plt.title('Log log graph')
arr = []
rv = poisson(y)
#print(len(g.node))
for n in range(max(l)):
    
    arr.append(rv.pmf(n))
   
    
#print(len(g.node)*rv.pmf(1),10)
a=[]
for n in l:
    n=math.log(n,10)
    a.append(n)
plt.grid(True)
plt.plot(arr)'''
    
