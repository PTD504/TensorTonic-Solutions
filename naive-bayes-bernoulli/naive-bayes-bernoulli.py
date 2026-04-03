import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    # Convert inputs to numpy arrays for internal processing
    X = np.array(X_train)
    y = np.array(y_train)
    X_t = np.array(X_test)
    
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]
    
    log_priors = np.zeros(n_classes)
    # Store log probabilities for feature being 1 (index 1) and 0 (index 0)
    log_likelihoods = np.zeros((n_classes, n_features, 2))
    
    # Training: Compute parameters using natural log
    for i, c in enumerate(classes):
        X_c = X[y == c]
        log_priors[i] = np.log(len(X_c) / len(X))
        
        # Apply Laplace smoothing to avoid zero probability
        counts = np.sum(X_c, axis=0)
        prob_1 = (counts + 1) / (len(X_c) + 2)
        log_likelihoods[i, :, 1] = np.log(prob_1)
        log_likelihoods[i, :, 0] = np.log(1 - prob_1)
        
    # Testing: Calculate log-likelihood for each test sample
    results = np.zeros((len(X_t), n_classes))
    for i in range(n_classes):
        # Select log probability based on whether feature is 0 or 1
        # Summing in log space is equivalent to multiplying in probability space
        log_lik_class = np.sum(np.where(X_t == 1, log_likelihoods[i, :, 1], log_likelihoods[i, :, 0]), axis=1)
        results[:, i] = log_priors[i] + log_lik_class
        
    return results