import numpy as np
import scipy
import sklearn
from sklearn import metrics
from six.moves import range
from sklearn.metrics import mean_squared_error

"""Unsupervised Metrics

Unsupervised scores based on code covariance and mutual information
    Gaussian total correlation.
    Gaussian Wasserstein correlation.
    Compute average mutual information between different factors.

Based on "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"
(https://arxiv.org/pdf/1811.12359.pdf).
"""

def generate_batch_factor_code(ground_truth_data, representation_function,
                                num_points, random_state, batch_size):
    """Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observation as input and
        outputs a representation.
        num_points: Number of points to sample.
        random_state: Numpy random state used for randomness.
        batch_size: Batchsize to sample points.

    Returns:
        representations: Codes (num_codes, num_points)-np array.
        factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = \
            ground_truth_data.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations,
                                        representation_function(
                                            current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)

# Gaussian total correlation
def gaussian_total_correlation(features):
    """Computes the total correlation of a Gaussian with covariance matrix cov.

    We use that the total correlation is the KL divergence between the Gaussian
    and the product of its marginals. By design, the means of these two Gaussians
    are zero and the covariance matrix of the second Gaussian is equal to the
    covariance matrix of the first Gaussian with off-diagonal entries set to zero.

    Args:
        features: Representation

    Returns:
        Scalar with total correlation.
    """
    cov = np.cov(features)
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])

def gaussian_wasserstein_correlation(features):
    """Wasserstein L2 distance between Gaussian and the product of its marginals.

    Args:
        features: Representation

    Returns:
        Scalar with score.
    """
    cov = np.cov(features)
    sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
    return 2 * np.trace(cov) - 2 * np.trace(sqrtm)

def gaussian_wasserstein_correlation_norm(features):
    """Wasserstein L2 distance between Gaussian and the product of its marginals.

    Args:
        features: Representation

    Returns:
        Scalar with score.
    """
    cov = np.cov(features)
    score = gaussian_wasserstein_correlation(cov)
    return score / np.sum(np.diag(cov))

  # Compute average mutual information between different factors.
def mutual_information_score(features):
    num_codes = features.shape[0]
    mus_discrete = histogram_discretize(features)
    mutual_info_matrix = discrete_mutual_info(mus_discrete, mus_discrete)
    np.fill_diagonal(mutual_info_matrix, 0)
    return np.sum(mutual_info_matrix) / (num_codes**2 - num_codes)

def histogram_discretize(features, num_bins=20):
    """
    Discretization based on histograms.
    """
    discretized = np.zeros_like(features)
    for i in range(features.shape[0]):
        discretized[i, :] = np.digitize(features[i, :], np.histogram(features[i, :], num_bins)[1][:-1])
    return discretized

def discrete_mutual_info(mus, ys):
    """
    Compute discrete mutual information.
    """
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


"""Supervised Metrics

Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""

def compute_mig(ground_truth_data, model, num_train=10000, batch_size=16):
    """Computes the mutual information gap.

    Args:
        features: Representation
        ground_truth_factor: Ground truth factors of variations
        num_train: Number of points used for training.

    Returns:
        Dict with average mutual information gap.
    """
    mus_train, ys_train = ground_truth_data.generate_batch_factor_code(model, num_train, batch_size)
    assert mus_train.shape[1] == num_train
    return _compute_mig(mus_train, ys_train)

def _compute_mig(mus_train, ys_train):
    """Computes score based on both training and testing codes and factors."""
    score_dict = {}
    discretized_mus = histogram_discretize(mus_train)
    m = discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    entropy = discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["discrete_mig"] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    return score_dict

def discrete_entropy(ys):
    """
    Compute discrete mutual information.
    """
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h


"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""

def compute_dci(ground_truth_data, model, num_train, num_test, batch_size=16, boost_mode="sklearn"):
    """Computes the DCI scores according to Sec 2.

    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        model: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
        num_train: Number of points used for training.
        num_test: Number of points used for testing.
        batch_size: Batch size for sampling.

    Returns:
        Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    # mus_train are of shape [num_codes, num_train], while ys_train are of shape
    # [num_factors, num_train].
    mus_train, ys_train = ground_truth_data.generate_batch_factor_code(model, num_train, batch_size)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    mus_test, ys_test = ground_truth_data.generate_batch_factor_code(model, num_test, batch_size)
    scores = _compute_dci(mus_train, ys_train, mus_test, ys_test, boost_mode)
    return scores

def _compute_dci(mus_train, ys_train, mus_test, ys_test, boost_mode):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test, boost_mode)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores

def compute_importance_gbt(x_train, y_train, x_test, y_test, boost_mode="sklearn"):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors])

    train_loss, test_loss  = [], []
    for i in range(num_factors):
        if boost_mode == "sklearn":
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier()
        elif boost_mode == "regressor":
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor()
        elif boost_mode == "xgboost":
            from xgboost import XGBClassifier

            model = XGBClassifier()
        elif boost_mode == "lightgbm":
            from lightgbm import LGBMClassifier

            model = LGBMClassifier()
        else:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier()

        model.fit(x_train.T, y_train[i, :])

        importance_matrix[:, i] = np.abs(model.feature_importances_)
        if boost_mode == "regressor":
            train_errors.append(mean_squared_error(model.predict(model_z_train), true_z_train[:, i]))
            test_errors.append(mean_squared_error(model.predict(model_z_test), true_z_test[:, i]))
        else:
            train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
            test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)

# noinspection PyUnresolvedReferences
def disentanglement(importance_matrix, eps=1e-11):
    """Computes the disentanglement score from an importance matrix"""

    disentanglement_per_code = _disentanglement_per_code(importance_matrix)

    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    disentanglement = np.sum(disentanglement_per_code * code_importance)
    return disentanglement

def _disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1.0 - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])

def completeness(importance_matrix, eps=1e-11):
    """Computes the completeness score from an importance matrix"""

    # noinspection PyUnresolvedReferences
    completeness_per_factor = _completeness_per_factor(importance_matrix) 

    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()

    completeness = np.sum(completeness_per_factor * factor_importance)
    return completeness

def _completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1.0 - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])



