# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Loss functions, metrics, training utilities """

import itertools

import torch
from torch import nn
import numpy as np
import scipy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

LOG_MEAN_VARS = {
    "elbo",
    "kl",
    "kl_epsilon",
    "kl_intervention_target",
    "mse",
    "consistency_mse",
    "inverse_consistency_mse",
    "log_prior",
    "log_prior_observed",
    "log_prior_intervened",
    "log_prior_nonintervened",
    "log_likelihood",
    "log_posterior",
    "z_regularization",
    "edges",
    "cyclicity",
    "encoder_std",
}


class VAEMetrics(nn.Module):
    """Metrics for generative training (maximizing the marginal likelihood / ELBO)"""

    def __init__(self, dim_z=2):
        super().__init__()
        self.dim_z = dim_z

    def forward(
        self,
        loss,
        true_intervention_labels=None,
        solution_std=None,
        intervention_posterior=None,
        eps=1.0e-9,
        z_regularization_amount=0.0,
        consistency_regularization_amount=0.0,
        inverse_consistency_regularization_amount=0.0,
        edge_regularization_amount=0.0,
        cyclicity_regularization_amount=0.0,
        intervention_entropy_regularization_amount=0.0,
        **model_outputs,
    ):
        metrics = {}
        batchsize = loss.shape[0]

        # beta-VAE loss
        loss = torch.mean(loss)

        # Regularization term
        loss = self._regulate(
            batchsize,
            consistency_regularization_amount,
            eps,
            intervention_entropy_regularization_amount,
            intervention_posterior,
            inverse_consistency_regularization_amount,
            loss,
            metrics,
            model_outputs,
            z_regularization_amount,
            edge_regularization_amount,
            cyclicity_regularization_amount,
        )

        assert torch.isfinite(loss)
        metrics["loss"] = loss.item()

        # Additional logged metrics (non-differentiable)
        with torch.no_grad():
            # Intervention posterior
            if true_intervention_labels is not None:
                self._evaluate_intervention_posterior(
                    eps, metrics, true_intervention_labels, intervention_posterior
                )

            # Mean std in p(epsilon2|epsilon1)
            if solution_std is not None:
                for i in range(solution_std.shape[-1]):
                    metrics[f"solution_std_{i}"] = torch.mean(solution_std[..., i]).item()

            # For most other quantities logged, just keep track of the mean
            for key in LOG_MEAN_VARS:
                if key in model_outputs:
                    try:
                        metrics[key] = torch.mean(model_outputs[key].to(torch.float)).item()
                    except AttributeError:
                        metrics[key] = float(model_outputs[key])

        return loss, metrics

    def _regulate(
        self,
        batchsize,
        consistency_regularization_amount,
        eps,
        intervention_entropy_regularization_amount,
        intervention_posterior,
        inverse_consistency_regularization_amount,
        loss,
        metrics,
        model_outputs,
        z_regularization_amount,
        edge_regularization_amount,
        cyclicity_regularization_amount,
    ):
        if edge_regularization_amount is not None and "edges" in model_outputs:
            loss += edge_regularization_amount * torch.mean(model_outputs["edges"])

        if cyclicity_regularization_amount is not None and "cyclicity" in model_outputs:
            try:
                loss += cyclicity_regularization_amount * torch.mean(model_outputs["cyclicity"])
            except TypeError:  # some models return a float
                loss += cyclicity_regularization_amount * model_outputs["cyclicity"]

        if z_regularization_amount is not None and "z_regularization" in model_outputs:
            loss += z_regularization_amount * torch.mean(model_outputs["z_regularization"])

        if consistency_regularization_amount is not None and "consistency_mse" in model_outputs:
            loss += consistency_regularization_amount * torch.mean(model_outputs["consistency_mse"])

        if (
            inverse_consistency_regularization_amount is not None
            and "inverse_consistency_mse" in model_outputs
        ):
            loss += inverse_consistency_regularization_amount * torch.mean(
                model_outputs["inverse_consistency_mse"]
            )

        if (
            inverse_consistency_regularization_amount is not None
            and "inverse_consistency_mse" in model_outputs
        ):
            loss += inverse_consistency_regularization_amount * torch.mean(
                model_outputs["inverse_consistency_mse"]
            )

        if (
            intervention_entropy_regularization_amount is not None
            and intervention_posterior is not None
        ):
            aggregate_posterior = torch.mean(intervention_posterior, dim=0)
            intervention_entropy = -torch.sum(
                aggregate_posterior * torch.log(aggregate_posterior + eps)
            )
            loss -= (
                intervention_entropy_regularization_amount * intervention_entropy
            )  # note minus sign: maximize entropy!
            metrics["intervention_entropy"] = intervention_entropy.item()

            # Let's also log the entropy corresponding to the determinstic (argmax) intervention
            # encoder
            most_likely_intervention = torch.argmax(intervention_posterior, dim=1)  # (batchsize,)
            det_posterior = torch.zeros_like(intervention_posterior)
            det_posterior[torch.arange(batchsize), most_likely_intervention] = 1.0
            aggregate_det_posterior = torch.mean(det_posterior, dim=0)
            det_intervention_entropy = -torch.sum(
                aggregate_det_posterior * torch.log(aggregate_det_posterior + eps)
            )
            metrics["intervention_entropy_deterministic"] = det_intervention_entropy.item()

        return loss

    @torch.no_grad()
    def _evaluate_intervention_posterior(
        self, eps, metrics, true_intervention_labels, intervention_posterior
    ):
        # We don't really want to iterate over all permutations of 32 latent variables
        if self.dim_z > 5:
            return

        # Some methods don't compute an intervention posterior
        if intervention_posterior is None:
            return

        batchsize = true_intervention_labels.shape[0]
        idx = torch.arange(batchsize)

        for i in range(intervention_posterior.shape[1]):
            metrics[f"intervention_posterior_{i}"] = torch.mean(intervention_posterior[:, i]).item()

        # Find all permutations of dim_z variables, and evaluate probability of true intervention
        # + accuracy
        true_int_prob, log_true_int_prob, int_accuracy = -float("inf"), -float("inf"), -float("inf")
        for permutation in itertools.permutations(list(range(1, self.dim_z + 1))):
            permutation = [0] + list(permutation)
            intervention_probs_permuted = intervention_posterior[:, permutation]
            predicted_intervention_permuted = torch.zeros_like(intervention_probs_permuted)
            predicted_intervention_permuted[
                idx, torch.argmax(intervention_probs_permuted, dim=1)
            ] = 1.0

            # log p(I*)
            log_true_int_prob_ = torch.mean(
                torch.log(
                    intervention_probs_permuted[idx, true_intervention_labels.flatten()] + eps
                )
            ).item()
            log_true_int_prob = max(log_true_int_prob, log_true_int_prob_)

            # p(I*)
            true_int_prob_ = torch.mean(
                intervention_probs_permuted[idx, true_intervention_labels.flatten()]
            ).item()
            true_int_prob = max(true_int_prob, true_int_prob_)

            # Accuracy
            int_accuracy_ = torch.mean(
                predicted_intervention_permuted[idx, true_intervention_labels.flatten()]
            ).item()
            int_accuracy = max(int_accuracy, int_accuracy_)

        metrics[f"intervention_correct_log_posterior"] = log_true_int_prob
        metrics[f"intervention_correct_posterior"] = true_int_prob
        metrics[f"intervention_accuracy"] = int_accuracy

def compute_dci(
    true_z_train, model_z_train, true_z_test, model_z_test, return_full_importance_matrix=False
):
    """
    Computes the DCI scores (disentanglement, completeness, informativeness) from a given dataset.

    Based on Eastwood & Williams, ICLR 2018 (https://openreview.net/pdf?id=By-7dz-AZ).

    Parameters:
    -----------
    true_z_train : torch.Tensor, shape (n_samples, n_true_latents)
    model_z_train : torch.Tensor, shape (n_samples, n_true_latents)
    true_z_test : torch.Tensor, shape (n_samples, n_model_latents)
    model_z_test : torch.Tensor, shape (n_samples, n_model_latents)

    Returns:
    --------
    results : dict
    """

    # Check inputs and convert to numpy arrays
    _verify_inputs(true_z_train, model_z_train, true_z_test, model_z_test)
    model_z_test = model_z_test.detach().cpu().data.numpy()
    model_z_train = model_z_train.detach().cpu().data.numpy()
    true_z_test = true_z_test.detach().cpu().data.numpy()
    true_z_train = true_z_train.detach().cpu().data.numpy()

    # Train classifier and compute importance matrix
    importance_matrix, train_err, test_err = _train_dci_classifier(
        true_z_train, model_z_train, true_z_test, model_z_test
    )

    # Extract DCI metrics
    metrics = {
        "informativeness_train": train_err,
        "informativeness_test": test_err,
        "disentanglement": _compute_disentanglement(importance_matrix),
        "completeness": _compute_completeness(importance_matrix),
    }

    # Optionally, also return full importance matrix
    if return_full_importance_matrix:
        for i in range(importance_matrix.shape[0]):
            for j in range(importance_matrix.shape[1]):
                metrics[f"importance_matrix_{i}_{j}"] = importance_matrix[i, j]

    return metrics


def _verify_inputs(true_z_train, model_z_train, true_z_test, model_z_test):
    assert (
        len(true_z_train.shape)
        == len(model_z_train.shape)
        == len(true_z_test.shape)
        == len(model_z_test.shape)
        == 2
    )
    batchsize_train, dim_z_true = true_z_train.shape
    batchsize_test, dim_z_model = model_z_test.shape
    assert true_z_test.shape == (batchsize_test, dim_z_true)
    assert model_z_train.shape == (batchsize_train, dim_z_model)


def _train_dci_classifier(true_z_train, model_z_train, true_z_test, model_z_test):
    """
    Trains a boosted decision tree to predict true factors of variation from model latents, and
    returns importance matrix
    """

    _, dim_z_true = true_z_train.shape
    _, dim_z_model = model_z_test.shape

    importance_matrix = np.zeros(shape=[dim_z_model, dim_z_true])

    # Loop over true factors of variation and train a predictor each
    train_errors, test_errors = [], []
    for i in range(dim_z_true):
        model = GradientBoostingRegressor()
        model.fit(model_z_train, true_z_train[:, i])

        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_errors.append(mean_squared_error(model.predict(model_z_train), true_z_train[:, i]))
        test_errors.append(mean_squared_error(model.predict(model_z_test), true_z_test[:, i]))

    return importance_matrix, np.mean(train_errors), np.mean(test_errors)


# noinspection PyUnresolvedReferences
def _compute_disentanglement(importance_matrix, eps=1.0e-12):
    """Computes the disentanglement score from an importance matrix"""

    disentanglement_per_code = 1.0 - scipy.stats.entropy(
        importance_matrix.T + eps, base=importance_matrix.shape[1]
    )

    if np.abs(importance_matrix.sum()) < eps:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    disentanglement = np.sum(disentanglement_per_code * code_importance)
    return disentanglement


def _compute_completeness(importance_matrix, eps=1.0e-12):
    """Computes the completeness score from an importance matrix"""

    # noinspection PyUnresolvedReferences
    completeness_per_factor = 1.0 - scipy.stats.entropy(
        importance_matrix + eps, base=importance_matrix.shape[0]
    )

    if np.abs(importance_matrix.sum()) < eps:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()

    completeness = np.sum(completeness_per_factor * factor_importance)
    return completeness