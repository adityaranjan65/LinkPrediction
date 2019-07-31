def node2vec_scores(
    g_train, train_test_split,
    P = 1, # Return hyperparameter
    Q = 1, # In-out hyperparameter
    WINDOW_SIZE = 10, # Context size for optimization
    NUM_WALKS = 10, # Number of walks per source
    WALK_LENGTH = 80, # Length of walk per source
    DIMENSIONS = 128, # Embedding dimension
    DIRECTED = False, # Graph directed/undirected
    WORKERS = 8, # Num. parallel workers
    ITER = 1, # SGD epochs
    edge_score_mode = "edge-emb", # Whether to use bootstrapped edge embeddings + LogReg (like in node2vec paper), 
        # or simple dot-product (like in GAE paper) for edge scoring
    verbose=1,
    ):
    if g_train.is_directed():
        DIRECTED = True

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    # Preprocessing, generate walks
    if verbose >= 1:
        print ('Preprocessing grpah for node2vec...')
    g_n2v = node2vec.Graph(g_train, DIRECTED, P, Q) # create node2vec graph instance
    g_n2v.preprocess_transition_probs()
    if verbose == 2:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=True)
    else:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=False)
    walks = [list(map(str, walk)) for walk in walks]

    # Train skip-gram model
    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

    # Store embeddings mapping
    emb_mappings = model.wv

    # Create node embeddings matrix (rows = nodes, columns = embedding features)
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
        # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if edge_score_mode == "edge-emb":
        
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        if len(val_edges) > 0:
            n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        # Test set scores
        n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print ("Invalid edge_score_mode! Either use edge-emb or dot-product.")

    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores
node2vec_scores
