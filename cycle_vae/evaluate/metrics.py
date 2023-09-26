import numpy as np
import scipy
from sklearn import metrics

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

def histogram_discretize(features, num_bins=2):
    """Discretization based on histograms."""
    discretized = np.zeros_like(features)
    for i in range(features.shape[0]):
        discretized[i, :] = np.digitize(features[i, :], np.histogram(features[i, :], num_bins)[1][:-1])
    return discretized

def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


"""Supervised Metrics

Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""

def compute_mig(features, ground_truth_factor, num_train=50):
    """Computes the mutual information gap.

    Args:
        features: Representation
        ground_truth_factor: Ground truth factors of variations
        num_train: Number of points used for training.

    Returns:
        Dict with average mutual information gap.
    """
    assert features.shape[1] == num_train
    return _compute_mig(features, ground_truth_factor)

def compute_mig_on_fixed_data(observations, labels, representation_function,
                              batch_size=100):
  """Computes the MIG scores on the fixed set of observations and labels.

  Args:
    observations: Observations on which to compute the score. Observations have
      shape (num_observations, 64, 64, num_channels).
    labels: Observed factors of variations.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Batch size used to compute the representation.

  Returns:
    MIG computed on the provided observations and labels.
  """
  mus = obtain_representation(observations, representation_function,
                                    batch_size)
  assert labels.shape[1] == observations.shape[0], "Wrong labels shape."
  assert mus.shape[1] == observations.shape[0], "Wrong representation shape."
  return _compute_mig(mus, labels)

def _compute_mig(mus_train, ys_train):
    """Computes score based on both training and testing codes and factors."""
    score_dict = {}
    discretized_mus = histogram_discretize(mus_train)
    m = discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    entropy = discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]
    return np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))

def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h