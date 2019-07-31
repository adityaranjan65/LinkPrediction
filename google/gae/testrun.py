from __future__ import division
from __future__ import print_function
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
# Convert features from normal matrix --> sparse matrix --> tuple
    # features_tuple contains: (list of matrix coordinates, list of values, matrix dimensions)
import scipy.sparse as sp
import time
import os
EGO_USER = 100466178325794757407# which ego network to look at

# Load pickled (adj, feat) tuple
network_dir = './g-processed/{0}-adj-feat.pkl'.format(EGO_USER)
with open(network_dir, 'rb') as f:
    adj, features = pickle.load(f,encoding='iso-8859-1')
g = nx.Graph(adj)
nx.draw_networkx(g, with_labels=False, node_size=50, node_color='r')
plt.show()



# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""


x = sp.lil_matrix(features)
features_tuple = sparse_to_tuple(x)
features_shape = features_tuple[2]
# Get graph attributes (to feed into model)
num_nodes = adj.shape[0] # number of nodes in adjacency matrix
num_features = features_shape[1] # number of features (columsn of features matrix)
features_nonzero = features_tuple[1].shape[0] # number of non-zero entries in features matrix (or length of values list)
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros() 
np.random.seed(0) # IMPORTANT: guarantees consistent train/test splits
adj_train, train_edges, train_edges_false, val_edges, val_edges_false,test_edges, test_edges_false = mask_test_edges(adj, test_frac=.3, val_frac=.1)

# Normalize adjacency matrix
adj_norm = preprocess_graph(adj_train)

# Add in diagonals
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
# Inspect train/test split
print("Total nodes:", adj.shape[0])
print("Total edges:", int(adj.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print("Training edges (positive):", len(train_edges))
print("Training edges (negative):", len(train_edges_false))
print("Validation edges (positive):", len(val_edges))
print("Validation edges (negative):", len(val_edges_false))
print("Test edges (positive):", len(test_edges))
print("Test edges (negative):", len(test_edges_false))
# Define hyperparameters
LEARNING_RATE = 0.005
EPOCHS = 300
HIDDEN1_DIM = 32
HIDDEN2_DIM = 16
DROPOUT = 0.1
# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32,shape=(None,3344)),
    'adj': tf.sparse_placeholder(tf.float32,shape=(None,3344)),
    'adj_orig': tf.sparse_placeholder(tf.float32,shape=(None,3344)),
    'dropout': tf.placeholder_with_default(0., shape=())
}
# How much to weigh positive examples (true edges) in cost print_function
  # Want to weigh less-frequent classes higher, so as to prevent model output bias
  # pos_weight = (num. negative samples / (num. positive samples)
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

# normalize (scale) average weighted cost
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
# Create VAE model
model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero,
                   HIDDEN1_DIM, HIDDEN2_DIM)

opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           learning_rate=LEARNING_RATE)
print(opt)
# Calculate ROC AUC
def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]])) # predicted score for given edge
        pos.append(adj_orig[e[0], e[1]]) # actual value (1)

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]])) # predicted score for given edge
        neg.append(adj_orig[e[0], e[1]]) # actual value (0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    
    
    ap_score = average_precision_score(labels_all, preds_all)
    
    precision, recall, thresholds = precision_recall_curve(labels_all, preds_all)
    
    '''fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')'''
    return roc_score, ap_score,precision,recall
def get_roc_scor(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]])) # predicted score for given edge
        pos.append(adj_orig[e[0], e[1]]) # actual value (1)

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]])) # predicted score for given edge
        neg.append(adj_orig[e[0], e[1]]) # actual value (0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    
    
    ap_score = average_precision_score(labels_all, preds_all)
    
    precision, recall, thresholds = precision_recall_curve(labels_all, preds_all)
    
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    #plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
    #plt.plot(fpr, tpr, marker='.',label='GAE')
    plt.plot(recall,precision,marker='.',label='GAE')
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRECISION RECALL CURVES')
  
    return roc_score, ap_score,precision,recall


# show the plot
plt.show()
cost_val = []
acc_val = []
val_roc_score = []

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
a=[]
b=[]
c=[]
d=[]
e=[]
# Train model

for epoch in range(EPOCHS):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features_tuple, placeholders)
    feed_dict.update({placeholders['dropout']: DROPOUT})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    # Evaluate predictions
    roc_curr, ap_curr,precision,recall = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)
    
    a.append(avg_cost)
    b.append(avg_accuracy)
    c.append(val_roc_score[-1])
    d.append(ap_curr)
    e.append(time.time() - t)
    # Print results for this epoch
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")
#print(a,b,c,d,e)
# Print final results
roc_score, ap_score,precision,recall = get_roc_scor(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
print(len(recall))
print('precision'+str(sum(precision)/len(precision)))
print('recall'+str(sum(recall)/len(recall)))
'''fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')'''


# coding: utf-8

# # Link Prediction Baselines
# ---
# ROC AUC and Average Precision computed on Facebook dataset using these link prediction baselines:
# 1. Adamic-Adar
# 2. Jaccard Coefficient
# 3. Preferential Attachment

# ## 1. Read in Graph Data

# In[2]:
# which ego network to look at

# Load pickled (adj, feat) tuple
network_dir = './g-processed/{0}-adj-feat.pkl'.format(EGO_USER)
with open(network_dir, 'rb') as f:
    adj, features = pickle.load(f)
    
g = nx.Graph(adj)


# In[3]:


# draw network
#nx.draw_networkx(g, with_labels=False, node_size=50, node_color='r')
#plt.show()


# ## 2. Preprocessing/Train-Test Split

# In[4]:


from preprocessing import mask_test_edges
np.random.seed(0) # make sure train-test split is consistent between notebooks
adj_sparse = nx.to_scipy_sparse_matrix(g)

# Perform train-test split
#adj_train, train_edges, train_edges_false, val_edges, val_edges_false,     test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.1)
g_train = nx.from_scipy_sparse_matrix(adj_train) # new graph object with only non-hidden edges


# In[5]:


# Inspect train/test split
print ("Total nodes:", adj_sparse.shape[0])
print ("Total edges:", int(adj_sparse.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print ("Training edges (positive):", len(train_edges))
print ("Training edges (negative):", len(train_edges_false))
print ("Validation edges (positive):", len(val_edges))
print ("Validation edges (negative):", len(val_edges_false))
print ("Test edges (positive):", len(test_edges))
print ("Test edges (negative):", len(test_edges_false))


# In[6]:


def get_roc_score1(edges_pos, edges_neg, score_matrix):
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]]) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]]) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    
    precision, recall, thresholds = precision_recall_curve(labels_all, preds_all)
    
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    #plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
    if(k==0):
        leg="ADAMIC-ADAR"
    elif(k==1):
        leg="JACCARD"
    elif(k==2):
        leg="PREFERENTIAL ATTAT."
    
    #plt.plot(fpr, tpr, marker='.',label=leg)
    
    plt.plot(recall,precision,marker='.',label=leg)
    plt.legend()
    return roc_score, ap_score,precision,recall


# ## 3. Adamic-Adar

# In[7]:


# Compute Adamic-Adar indexes from g_train
aa_matrix = np.zeros(adj.shape)
for u, v, p in nx.adamic_adar_index(g_train): # (u, v) = node indices, p = Adamic-Adar index
    aa_matrix[u][v] = p
    aa_matrix[v][u] = p # make sure it's symmetric
    
# Normalize array
aa_matrix = aa_matrix / aa_matrix.max()


# In[8]:


# Calculate ROC AUC and Average Precision
k=0
aa_roc, aa_ap,precision,recall= get_roc_score1(test_edges, test_edges_false, aa_matrix)

print ('Adamic-Adar Test ROC score: ', str(aa_roc))
print ('Adamic-Adar Test AP score: ', str(aa_ap))
print('precision'+str(sum(precision)/len(precision)))
print('recall'+str(sum(recall)/len(recall)))


# ## 4. Jaccard Coefficient

# In[9]:


# Compute Jaccard Coefficients from g_train
jc_matrix = np.zeros(adj.shape)
for u, v, p in nx.jaccard_coefficient(g_train): # (u, v) = node indices, p = Jaccard coefficient
    jc_matrix[u][v] = p
    jc_matrix[v][u] = p # make sure it's symmetric
    
# Normalize array
jc_matrix = jc_matrix / jc_matrix.max()


# In[10]:

precision=[]
recall=[]
# Calculate ROC AUC and Average Precision
k=1
jc_roc, jc_ap,precision,recall = get_roc_score1(test_edges, test_edges_false, jc_matrix)

print ('Jaccard Coefficient Test ROC score: ', str(jc_roc))
print ('Jaccard Coefficient Test AP score: ', str(jc_ap))
print('precision'+str(sum(precision)/len(precision)))
print('recall'+str(sum(recall)/len(recall)))

# ## 5. Preferential Attachment

# In[11]:


# Calculate, store Adamic-Index scores in array
pa_matrix = np.zeros(adj.shape)
for u, v, p in nx.preferential_attachment(g_train): # (u, v) = node indices, p = Jaccard coefficient
    pa_matrix[u][v] = p
    pa_matrix[v][u] = p # make sure it's symmetric
    
# Normalize array
pa_matrix = pa_matrix / pa_matrix.max()


# In[12]:


# Calculate ROC AUC and Average Precision
precision=[]
recall=[]
k=2
pa_roc, pa_ap,precision,recall = get_roc_score1(test_edges, test_edges_false, pa_matrix)

print ('Preferential Attachment Test ROC score: ', str(pa_roc))
print ('Preferential Attachment Test AP score: ', str(pa_ap))
print('precision'+str(sum(precision)/len(precision)))
print('recall'+str(sum(recall)/len(recall)))
np.random.seed(0) # make sure train-test split is consistent between notebooks
adj_sparse = nx.to_scipy_sparse_matrix(g)

# Perform train-test split
#adj_train, train_edges, train_edges_false, val_edges, val_edges_false,     test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.1)
g_train = nx.from_scipy_sparse_matrix(adj_train) # new graph object with only non-hidden edges


# In[5]:


# Inspect train/test split
print( "Total nodes:", adj_sparse.shape[0])
print( "Total edges:", int(adj_sparse.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print ("Training edges (positive):", len(train_edges))
print ("Training edges (negative):", len(train_edges_false))
print ("Validation edges (positive):", len(val_edges))
print ("Validation edges (negative):", len(val_edges_false))
print ("Test edges (positive):", len(test_edges))
print ("Test edges (negative):", len(test_edges_false))


# In[6]:


def get_roc_score2(edges_pos, edges_neg, embeddings):
    score_matrix = np.dot(embeddings, embeddings.T)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    precision, recall, thresholds = precision_recall_curve(labels_all, preds_all)
    
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    #plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
    #plt.plot(fpr, tpr, marker='.',label="SPECTRAL CLUSTERING")
    
    plt.plot(recall,precision,marker='.',label="SPECTRAL CLUSTERING")
    plt.legend()
    return roc_score, ap_score,precision,recall

# ## 3. Spectral Clustering

# In[7]:


from sklearn.manifold import spectral_embedding

# Get spectral embeddings (16-dim)
emb = spectral_embedding(adj_train, n_components=16, random_state=0)


# In[8]:

precision=[]
recall=[]
# Calculate ROC AUC and Average Precision
sc_roc, sc_ap,precision,recall = get_roc_score2(test_edges, test_edges_false, emb)

print ('Spectral Clustering Test ROC score: ', str(sc_roc))
print ('Spectral Clustering Test AP score: ', str(sc_ap))
print('precision'+str(sum(precision)/len(precision)))
print('recall'+str(sum(recall)/len(recall)))
