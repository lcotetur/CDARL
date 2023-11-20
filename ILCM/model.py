import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain
from collections import defaultdict

import nflows.transforms
import nflows.utils
import nflows.distributions

from CDARL.ILCM.nets import make_mlp, make_elementwise_mlp
from CDARL.ILCM.experiment_utils import mask, clean_and_clamp

import itertools
import numpy as np

DEFAULT_BASE_DENSITY = nflows.distributions.StandardNormal((1,))

def logmeanexp(x, dim):
    """Like logsumexp, but using a mean instead of the sum"""
    return torch.logsumexp(x, dim=dim) - np.log(x.shape[dim])

class ImplicitSCM(nn.Module):
    """
    Implicit causal model, centered around noise encoding and solution functions.

    Parameters:
    -----------
    graph: Graph or None
        Causal graph. If None, the full graph is assumed (no masking)
    solution_functions: list of self.dim_z Transforms
        The i-th element in this list is a diffeo that models `p(e'_i|e)` with noise encodings `e`
        like a flow
    """

    def __init__(
        self, graph, solution_functions, base_density, manifold_thickness, dim_z, causal_structure
    ):
        super().__init__()
        self.dim_z = dim_z

        self.solution_functions = torch.nn.ModuleList(solution_functions)
        self.base_density = base_density
        self.register_buffer("_manifold_thickness", torch.tensor(manifold_thickness))
        self.register_buffer("_mask_values", torch.zeros(dim_z))
        self.register_buffer("topological_order", torch.zeros(dim_z, dtype=torch.long))

        self.set_causal_structure(graph, causal_structure)

    def sample_noise_weakly_supervised(self, n, intervention, adjacency_matrix=None):
        """Samples in the weakly supervised setting for a given intervention"""

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, n)

        # Sample pre-intervention noise encodings
        epsilon1 = self._sample_noise(n)  # noise variables used for the data pre intervention

        # Sample intervention process for targets
        intervention_noise = self._sample_noise(n)  # noise used for the intervened-upon variables
        epsilon2 = (
            intervention
            * self._inverse(intervention_noise, epsilon1, adjacency_matrix=adjacency_matrix)[0]
        )

        # Counterfactual consistency noise for non-intervened variables
        cf_noise = self._sample_noise(n, True)  # noise used for the non-intervened-upon variables
        epsilon2 += (1.0 - intervention) * (epsilon1 + cf_noise)

        return epsilon1, epsilon2

    def log_prob_weakly_supervised(self, z1, z2, intervention, adjacency_matrix):
        """
        Given weakly supervised causal variables and the intervention mask, computes the
        corresponding noise variables and log likelihoods.
        """

        raise NotImplementedError

    def log_prob_noise_weakly_supervised(
        self,
        epsilon1,
        epsilon2,
        intervention,
        adjacency_matrix,
        include_intervened=True,
        include_nonintervened=True,
    ):
        """
        Given weakly supervised as noise encodings epsilon1, epsilon2 and the intervention mask,
        computes the corresponding causal variables and log likelihoods.
        """

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, epsilon1.shape[0])
        assert torch.all(torch.isfinite(epsilon1))
        assert torch.all(torch.isfinite(epsilon2))

        # Observed likelihood
        logprob_observed = self._compute_logprob_observed(epsilon1)
        logprob = logprob_observed

        # Intervention likelihood
        if include_intervened:
            log_det, logprob_intervened = self._compute_logprob_intervened(
                adjacency_matrix, epsilon1, epsilon2, intervention
            )
            logprob = logprob + logprob_intervened
        else:
            logprob_intervened = torch.zeros_like(logprob_observed)
            log_det = torch.zeros((epsilon1.shape[0], 1), device=epsilon1.device)

        # Counterfactual discrepancy for not-intervened-upon variables
        if include_nonintervened:
            logprob_nonintervened = self._compute_logprob_nonintervened(
                epsilon1, epsilon2, intervention
            )
            logprob = logprob + logprob_nonintervened
        else:
            logprob_nonintervened = torch.zeros_like(logprob_intervened)

        # Package outputs
        assert torch.all(torch.isfinite(logprob))
        outputs = dict(
            log_prior_observed=logprob_observed,
            log_prior_intervened=logprob_intervened,
            log_prior_nonintervened=logprob_nonintervened,
            solution_std=torch.exp(
                -log_det
            ),  # log_det is log(std) from noise encoding -> z transform
        )

        return logprob, outputs

    def _compute_logprob_nonintervened(self, epsilon1, epsilon2, intervention):
        cf_noise = (epsilon2 - epsilon1) / self.manifold_thickness
        assert torch.all(torch.isfinite(cf_noise))
        logprob_nonintervened = self.base_density.log_prob(cf_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_nonintervened -= torch.log(self.manifold_thickness)
        logprob_nonintervened = clean_and_clamp(logprob_nonintervened)
        logprob_nonintervened = (
            1.0 - intervention
        ) * logprob_nonintervened  # (batchsize, self.dim_z)
        logprob_nonintervened = torch.sum(logprob_nonintervened, 1, keepdim=True)  # (batchsize, 1)
        return logprob_nonintervened

    def _compute_logprob_intervened(self, adjacency_matrix, epsilon1, epsilon2, intervention):
        z_intervened, log_det = self._solve(
            epsilon=epsilon2, conditioning_epsilon=epsilon1, adjacency_matrix=adjacency_matrix
        )
        assert torch.all(torch.isfinite(z_intervened))
        logprob_intervened = self.base_density.log_prob(z_intervened.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_intervened += log_det
        logprob_intervened = intervention * logprob_intervened  # (batchsize, self.dim_z)
        logprob_intervened = clean_and_clamp(logprob_intervened)
        logprob_intervened = torch.sum(logprob_intervened, 1, keepdim=True)  # (batchsize, 1)
        return log_det, logprob_intervened

    def _compute_logprob_observed(self, epsilon1):
        logprob_observed = self.base_density.log_prob(epsilon1.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_observed = clean_and_clamp(logprob_observed)
        logprob_observed = torch.sum(logprob_observed, 1, keepdim=True)  # (batchsize, 1)
        return logprob_observed

    def noise_to_causal(self, epsilon, adjacency_matrix=None):
        """Given noise encoding, returns causal encoding"""

        return self._solve(epsilon, epsilon, adjacency_matrix=adjacency_matrix)[0]

    def causal_to_noise(self, z, adjacency_matrix=None):
        """Given causal latents, returns noise encoding"""

        assert self.topological_order is not None

        conditioning_epsilon = z.clone()
        epsilons = {}

        for i in self.topological_order:
            i = i.item()

            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)
            epsilon, _ = self.solution_functions[i](z[:, i : i + 1], context=masked_epsilon)

            epsilons[i] = epsilon
            conditioning_epsilon[:, i : i + 1] = epsilon

        epsilon = torch.cat([epsilons[i] for i in range(self.dim_z)], 1)

        return epsilon

    @property
    def manifold_thickness(self):
        """Returns counterfactual manifold thickness (only here for legacy reasons)"""
        return self._manifold_thickness

    @manifold_thickness.setter
    @torch.no_grad()
    def manifold_thickness(self, value):
        """Sets counterfactual manifold thickness (only here for legacy reasons)"""
        self._manifold_thickness.copy_(torch.as_tensor(value).to(self._manifold_thickness.device))

    @torch.no_grad()
    def get_scm_parameters(self):
        """Returns key parameters of causal model for logging purposes"""
        # Manifold thickness
        parameters = {"manifold_thickness": self.manifold_thickness}

        return parameters

    def generate_similar_intervention(
        self, z1, z2_example, intervention, adjacency_matrix, sharp_manifold=True
    ):
        """Infers intervention and "fakes" it in the model"""
        raise NotImplementedError

    @staticmethod
    def _sanitize_intervention(intervention, n):
        if intervention is not None:
            assert len(intervention.shape) == 2
            assert intervention.shape[0] == n
            intervention = intervention.to(torch.float)

        return intervention

    @torch.no_grad()
    def get_masked_solution_function(self, i):
        """Returns solution function where inputs are masked to conform to topological order"""
        return MaskedSolutionTransform(self, i)

    def _solve(self, epsilon, conditioning_epsilon, adjacency_matrix):
        """
        Given noise encodings, compute causal variables (= base variables in counterfactual flow).
        """

        zs = []
        logdets = []

        for i, transform in enumerate(self.solution_functions):
            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)

            z, logdet = transform.inverse(epsilon[:, i : i + 1], context=masked_epsilon)
            zs.append(z)
            logdets.append(logdet)

        z = torch.cat(zs, 1)
        logdet = torch.cat(logdets, 1)

        return z, logdet

    def _inverse(self, z, conditioning_epsilon, adjacency_matrix=None, order=None):
        if order is None:
            assert self.topological_order is not None
            order = self.topological_order

        epsilons = {}
        logdets = {}

        for i in order:
            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)

            epsilon, logdet = self.solution_functions[i](z[:, i : i + 1], context=masked_epsilon)
            epsilons[i] = epsilon
            logdets[i] = logdet

        epsilon = torch.cat([epsilons[i] for i in range(self.dim_z)], 1)
        logdet = torch.cat([logdets[i] for i in range(self.dim_z)], 1)

        return epsilon, logdet

    def get_masked_context(self, i, epsilon, adjacency_matrix):
        """Masks the input to a solution function to conform to topological order"""
        mask_ = self._get_ancestor_mask(
            i, adjacency_matrix, device=epsilon.device, n=epsilon.shape[0]
        )
        dummy_data = self._mask_values.unsqueeze(0)
        dummy_data[:, i] = 0.0
        masked_epsilon = mask(epsilon, mask_, mask_data=dummy_data)
        return masked_epsilon

    def _get_ancestor_mask(self, i, adjacency_matrix, device, n=1):
        if self.graph is None:
            if self.causal_structure == "fixed_order":
                ancestor_mask = torch.zeros((n, self.dim_z), device=device)
                ancestor_mask[..., self.ancestor_idx[i]] = 1.0
            elif self.causal_structure == "trivial":
                ancestor_mask = torch.zeros((n, self.dim_z), device=device)
            else:
                ancestor_mask = torch.ones((n, self.dim_z), device=device)
                ancestor_mask[..., i] = 0.0

        else:
            # Rather than the adjacency matrix, we're computing the
            # non-descendancy matrix: the probability of j not being a descendant of i
            # The idea is that this has to be gradient-friendly, soft adjacency-friendly way.
            # 1 - anc = (1 - adj) * (1 - adj^2) * (1 - adj^3) * ... * (1 - adj^(n-1))
            non_ancestor_matrix = torch.ones_like(adjacency_matrix)
            for n in range(1, self.dim_z):
                non_ancestor_matrix *= 1.0 - torch.linalg.matrix_power(adjacency_matrix, n)

            ancestor_mask = 1.0 - non_ancestor_matrix[..., i]

        return ancestor_mask

    def _sample_noise(self, n, sample_consistency_noise=False):
        """Samples noise"""
        if sample_consistency_noise:
            return self.manifold_thickness * self.base_density.sample(n * self.dim_z).reshape(
                n, self.dim_z
            )
        else:
            return self.base_density.sample(n * self.dim_z).reshape(n, self.dim_z)

    def set_causal_structure(
        self, graph, causal_structure, topological_order=None, mask_values=None
    ):
        """Fixes causal structure, usually to a given topoloigical order"""
        if graph is None:
            assert causal_structure in ["none", "fixed_order", "trivial"]

        if topological_order is None:
            topological_order = list(range(self.dim_z))

        if mask_values is None:
            mask_values = torch.zeros(self.dim_z, device=self._manifold_thickness.device)

        self.graph = graph
        self.causal_structure = causal_structure
        self.topological_order.copy_(torch.LongTensor(topological_order))
        self._mask_values.copy_(mask_values)

        self._compute_ancestors()

    def _compute_ancestors(self):
        # Construct ancestor index dict
        ancestor_idx = defaultdict(list)
        descendants = set(range(self.dim_z))
        for i in self.topological_order:
            i = i.item()
            descendants.remove(i)
            for j in descendants:
                ancestor_idx[j].append(i)

        self.ancestor_idx = ancestor_idx

    def load_state_dict(self, state_dict, strict=True):
        """Overloading the state dict loading so we can compute ancestor structure"""
        super().load_state_dict(state_dict, strict)
        self._compute_ancestors()



class MLPImplicitSCM(ImplicitSCM):
    """MLP-based implementation of ILCMs"""

    def __init__(self, graph_parameterization, manifold_thickness, dim_z, hidden_layers=1, hidden_units=100, base_density=DEFAULT_BASE_DENSITY, homoskedastic=True, min_std=None):
        solution_functions = []

        # Initialize graph
        graph = None
        causal_structure = "none"

        # Initialize transforms for p(e'|e)
        for _ in range(dim_z):
            solution_functions.append(
                make_mlp_structure_transform(
                    dim_z,
                    hidden_layers,
                    hidden_units,
                    homoskedastic,
                    min_std=min_std,
                    initialization="broad",
                )
            )

        super().__init__(graph, solution_functions, base_density, manifold_thickness, dim_z=dim_z, causal_structure=causal_structure)

def make_mlp_structure_transform(
    dim_z,
    hidden_layers,
    hidden_units,
    homoskedastic,
    min_std,
    concat_masks_to_parents=True,
    initialization = "broad",
):
    """
    Utility function that constructs an invertible transformation for causal mechanisms
    in SCMs
    """
    input_factor = 2 if concat_masks_to_parents else 1
    features = (
        [input_factor * dim_z]
        + [hidden_units for _ in range(hidden_layers)]
        + [1 if homoskedastic else 2]
    )
    param_net = make_mlp(features)

    #initialization == "broad":
    # For noise-centric models we want that the typical initial standard deviation in p(e2 | e1)
    # is large, around 10
    mean_bias_std = 1.0e-3
    mean_weight_std = 0.1

    last_layer = list(param_net._modules.values())[-1]
    #homoskedastic:
    nn.init.normal_(last_layer.bias, mean=0.0, std=mean_bias_std)
    nn.init.normal_(last_layer.weight, mean=0.0, std=mean_weight_std)

    structure_trf = ConditionalAffineScalarTransform(
        param_net=param_net, features=1, conditional_std=not homoskedastic, min_scale=min_std
    )

    return structure_trf

class ConditionalAffineScalarTransform(nflows.transforms.Transform):
    """
    Computes X = X * scale(context) + shift(context), where (scale, shift) are given by
    param_net(context). param_net takes as input the context with shape (batchsize,
    context_features) or None, its output has to have shape (batchsize, 2).
    """

    def __init__(self, param_net=None, features=None, conditional_std=True, min_scale=None):
        super().__init__()

        self.conditional_std = conditional_std
        self.param_net = param_net

        if self.param_net is None:
            assert features is not None
            self.shift = torch.zeros(features)
            torch.nn.init.normal_(self.shift)
            self.shift = torch.nn.Parameter(self.shift)
        else:
            self.shift = None

        if self.param_net is None or not conditional_std:
            self.log_scale = torch.zeros(features)
            torch.nn.init.normal_(self.log_scale)
            self.log_scale = torch.nn.Parameter(self.log_scale)
        else:
            self.log_scale = None

        if min_scale is None:
            self.min_scale = None
        else:
            self.register_buffer("min_scale", torch.tensor(min_scale))

    def get_scale_and_shift(self, context):
        if self.param_net is None:
            shift = self.shift.unsqueeze(1)
            log_scale = self.log_scale.unsqueeze(1)
        elif not self.conditional_std:
            shift = self.param_net(context)
            log_scale = self.log_scale.unsqueeze(1)
        else:
            log_scale_and_shift = self.param_net(context)
            log_scale = log_scale_and_shift[:, 0].unsqueeze(1)
            shift = log_scale_and_shift[:, 1].unsqueeze(1)

        scale = torch.exp(log_scale)
        if self.min_scale is not None:
            scale = scale + self.min_scale

        num_dims = torch.prod(torch.tensor([1]), dtype=torch.float)
        logabsdet = torch.log(scale) * num_dims

        return scale, shift, logabsdet

    def forward(self, inputs, context=None):
        scale, shift, logabsdet = self.get_scale_and_shift(context)
        outputs = inputs * scale + shift
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        scale, shift, logabsdet = self.get_scale_and_shift(context)
        outputs = (inputs - shift) / scale
        return outputs, -logabsdet

class MaskedSolutionTransform(nn.Module):
    """Transform wrapper around the solution function in an SCM"""

    def __init__(self, scm, scm_component):
        super().__init__()
        self.scm = scm
        self.scm_component = scm_component

    def forward(self, inputs, context):
        masked_context = self.scm.get_masked_context(self.scm_component, epsilon=context, adjacency_matrix=None)
        return self.scm.solution_functions[self.scm_component](inputs, context=masked_context)

    def inverse(self, inputs, context):
        masked_context = self.scm.get_masked_context(self.scm_component, epsilon=context, adjacency_matrix=None)
        return self.scm.solution_functions[self.scm_component](inputs, context=masked_context)

def add_coords(x):
    """
    Adds coordinate encodings to a tensor.

    Parameters:
    -----------
    x: torch.Tensor of shape (b, c, h, w)
        Input tensor

    Returns:
    --------
    augmented_x: torch.Tensor of shape (b, c+2, h, w)
        Input tensor augmented with two new channels with positional encodings
    """

    b, c, h, w = x.shape
    coords_h = torch.linspace(-1, 1, h, device=x.device)[:, None].expand(b, 1, h, w)
    coords_w = torch.linspace(-1, 1, w, device=x.device).expand(b, 1, h, w)
    return torch.cat([x, coords_h, coords_w], 1)


class CoordConv2d(nn.Module):
    """
    Conv2d that adds coordinate encodings to the input
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(add_coords(x))

class ImageEncoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z
    """

    def __init__(
        self,
        in_features,
        out_features,
        in_resolution=64,
        hidden_features=64,
        batchnorm=True,
        batchnorm_epsilon=0.1,
        conv_class=nn.Conv2d,
        mlp_layers=0,
        mlp_hidden=64,
        min_std=0.0,
        elementwise_layers=1,
        elementwise_hidden=16,
        permutation=0,
    ):
        super().__init__()

        self.net = self._make_conv_net(
            batchnorm,
            batchnorm_epsilon,
            conv_class,
            hidden_features,
            in_features,
            mlp_hidden,
            mlp_layers,
            out_features,
            in_resolution,
        )

        hidden_units = [mlp_hidden] * mlp_layers + [2 * out_features]
        self.mlp = make_mlp(hidden_units, activation="leaky_relu", initial_activation="leaky_relu")
        self.register_buffer("min_std", torch.tensor(min_std))

        self.elementwise = make_elementwise_mlp(
            [elementwise_hidden] * elementwise_layers, activation="leaky_relu"
        )

        if permutation == 0:
            self.permutation = None
        else:
            self.permutation = generate_permutation(out_features, permutation, inverse=False)

    def _make_conv_net(
        self,
        batchnorm,
        batchnorm_epsilon,
        conv_class,
        hidden_features,
        in_features,
        mlp_hidden,
        mlp_layers,
        out_features,
        in_resolution,
    ):

        net_out_features = mlp_hidden if mlp_layers > 0 else 2 * out_features
        kwargs = {
            "batchnorm": batchnorm,
            "batchnorm_epsilon": batchnorm_epsilon,
            "conv_class": conv_class,
        }

        net = nn.Sequential(
                ResNetDown(in_features, hidden_features, **kwargs),
                ResNetDown(hidden_features, 2 * hidden_features, **kwargs),
                ResNetDown(2 * hidden_features, 4 * hidden_features, **kwargs),
                ResNetDown(4 * hidden_features, 8 * hidden_features, **kwargs),
                ResNetDown(8 * hidden_features, 8 * hidden_features, **kwargs),
                ResNetDown(8 * hidden_features, 8 * hidden_features, **kwargs),
                conv_class(8 * hidden_features, net_out_features, 1),
            )

        return net

    def forward(
        self,
        x,
        eval_likelihood_at=None,
        deterministic=False,
        return_mean=False,
        return_std=False,
        full=True,
        reduction="sum",
    ):
        """
        Encode image, returns Gaussian

        [b, in_channels, 64, 64] -> Gaussian over [b, out_features]
        See gaussian_encode for parameters and return type.
        """

        mean, std = self.mean_std(x)
        return gaussian_encode(
            mean,
            std,
            eval_likelihood_at,
            deterministic,
            return_mean=return_mean,
            return_std=return_std,
            full=full,
            reduction=reduction,
        )

    def mean_std(self, x):
        """Encode image, return mean and std"""
        hidden = self.net(x).squeeze(3).squeeze(2)
        hidden = self.mlp(hidden)
        mean, std = vector_to_gaussian(hidden, min_std=self.min_std)

        mean = self.elementwise(mean)

        if self.permutation is not None:
            mean = mean[:, self.permutation]
            std = std[:, self.permutation]

        return mean, std

    def freeze(self):
        """Freeze convolutional net and MLP, but not elementwise transformation"""
        for parameter in chain(self.mlp.parameters(), self.net.parameters()):
            parameter.requires_grad = False

    def freezable_parameters(self):
        """Returns parameters that should be frozen during training"""
        return chain(self.mlp.parameters(), self.net.parameters())

    def unfreezable_parameters(self):
        """Returns parameters that should not be frozen during training"""
        return self.elementwise.parameters()

class ResNetDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=2,
        batchnorm=True,
        batchnorm_epsilon=0.01,
        conv_class=nn.Conv2d,
    ):
        super(ResNetDown, self).__init__()

        self.conv1 = conv_class(in_features, out_features // 2, 3, padding=1)
        self.conv2 = conv_class(out_features // 2, out_features, 3, padding=1)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(out_features // 2, eps=batchnorm_epsilon)
            self.bn2 = nn.BatchNorm2d(out_features, eps=batchnorm_epsilon)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.point_wise = conv_class(in_features, out_features, 1)
        self.pool = nn.AvgPool2d(scale, scale)

    def forward(self, x):
        skip = self.point_wise(self.pool(x))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(self.conv2(x))

        x = F.leaky_relu(x + skip)
        return x

class ImageDecoder(nn.Module):
    """
    Decoder block
    """

    def __init__(
        self,
        in_features,
        out_features,
        out_resolution=64,
        hidden_features=64,
        batchnorm=True,
        batchnorm_epsilon=0.1,
        conv_class=nn.Conv2d,
        fix_std=False,
        min_std=1e-3,
        mlp_layers=2,
        mlp_hidden=64,
        elementwise_layers=1,
        elementwise_hidden=16,
        permutation=0,
    ):
        super().__init__()

        if permutation == 0:
            self.permutation = None
        else:
            self.permutation = generate_permutation(in_features, permutation, inverse=True)

        self.elementwise = make_elementwise_mlp(
            [elementwise_hidden] * elementwise_layers, activation="leaky_relu"
        )

        hidden_units = [in_features] + [mlp_hidden] * mlp_layers
        self.mlp = make_mlp(hidden_units, activation="leaky_relu", final_activation="leaky_relu")

        self.net = self._create_conv_net(
            batchnorm,
            batchnorm_epsilon,
            conv_class,
            fix_std,
            hidden_features,
            in_features,
            mlp_hidden,
            mlp_layers,
            out_features,
            out_resolution,
        )
        self.fix_std = fix_std
        self.register_buffer("min_std", torch.tensor(min_std))

    def _create_conv_net(
        self,
        batchnorm,
        batchnorm_epsilon,
        conv_class,
        fix_std,
        hidden_features,
        in_features,
        mlp_hidden,
        mlp_layers,
        out_features,
        out_resolution,
    ):
        net_in_features = mlp_hidden if mlp_layers > 0 else in_features
        feature_multiplier = 1 if fix_std else 2
        kwargs = {
            "batchnorm": batchnorm,
            "batchnorm_epsilon": batchnorm_epsilon,
            "conv_class": conv_class,
        }

        net = nn.Sequential(
                ResNetUp(net_in_features, hidden_features * 8, **kwargs),
                ResNetUp(hidden_features * 8, hidden_features * 8, **kwargs),
                ResNetUp(hidden_features * 8, hidden_features * 4, **kwargs),
                ResNetUp(hidden_features * 4, hidden_features * 2, **kwargs),
                ResNetUp(hidden_features * 2, hidden_features, **kwargs),
                ResNetUp(hidden_features, hidden_features // 2, **kwargs),
                conv_class(hidden_features // 2, feature_multiplier * out_features, 1),
            )

        return net

    def forward(self, x, eval_likelihood_at=None, deterministic=False, return_mean=False, return_std=False, full=True, reduction="sum"):
        """
        Decodes latent into image, returns Gaussian

        [b, in_channels] -> Gaussian over [b, out_features, 64, 64]
        See gaussian_encode for parameters and return type.
        """

        mean, std = self.mean_std(x)
        return gaussian_encode(mean, std, eval_likelihood_at, deterministic, return_mean=return_mean, return_std=return_std, full=full, reduction=reduction)

    def mean_std(self, x):
        """Given latent, compute mean and std"""
        if self.permutation is not None:
            x = x[:, self.permutation]

        hidden = self.elementwise(x)
        hidden = self.mlp(hidden)
        hidden = self.net(hidden[:, :, None, None])
        mean, std = vector_to_gaussian(hidden, fix_std=self.fix_std, min_std=self.min_std)
        return mean, std

    def freezable_parameters(self):
        """Returns parameters that should be frozen during training"""
        return chain(self.mlp.parameters(), self.net.parameters())

    def unfreezable_parameters(self):
        """Returns parameters that should not be frozen during training"""
        return self.elementwise.parameters()

    def freeze(self):
        """Freeze convolutional net and MLP, but not elementwise transformation"""
        for parameter in chain(self.mlp.parameters(), self.net.parameters()):
            parameter.requires_grad = False


class ResNetUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=2,
        batchnorm=True,
        batchnorm_epsilon=0.01,
        conv_class=nn.Conv2d,
    ):
        super(ResNetUp, self).__init__()

        self.conv1 = conv_class(in_features, out_features // 2, 3, padding=1)
        self.conv2 = conv_class(out_features // 2, out_features, 3, padding=1)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(out_features // 2, eps=batchnorm_epsilon)
            self.bn2 = nn.BatchNorm2d(out_features, eps=batchnorm_epsilon)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.point_wise = conv_class(in_features, out_features, 1)
        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, x):
        skip = self.point_wise(self.upsample(x))

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.upsample(x)
        x = self.bn2(self.conv2(x))

        x = F.leaky_relu(x + skip)
        return x

def gaussian_encode(mean, std, eval_likelihood_at=None, deterministic=False, return_mean=False, return_std=False, full=True, reduction="sum"):
    """ 
    Given mean and std of Gaussian, compute likelihoods and sample.

    In an encoder: takes as input the observed data x and returns the latent representation inputs.
    In a decoder: takes as input the latent representation inputs and returns the reconstructed data
    x.

    Parameters:
    -----------
    mean : torch.Tensor with shape (batchsize, input_features), dtype torch.float
    std : torch.Tensor with shape (batchsize, input_features), dtype torch.float

    Returns:
    --------
    outputs : torch.Tensor with shape (batchsize, output_features), dtype torch.float
        Encoded or decoded version of the data
    log_likelihood : torch.Tensor with shape (batchsize, output_features), dtype torch.float
        Log likelihood evaluated at eval_likelihood_at or at outputs.
    encoder_std : torh.Tensor, optional
        If `return_std` is True, returns the encoder std
    """
    # Sample inputs via reparameterization trick and compute log likelihood
    if deterministic:
        z = mean
    else:
        u = torch.randn_like(mean)
        z = mean + std * u

    # Compute log likelihood
    if eval_likelihood_at is None:
        log_likelihood = gaussian_log_likelihood(z, mean, std, full=full, reduction=reduction)
    else:
        log_likelihood = gaussian_log_likelihood(
            eval_likelihood_at, mean, std, full=full, reduction=reduction
        )

    # Package results
    results = [z, log_likelihood]
    if return_mean:
        results.append(mean)
    if return_std:
        results.append(std)

    return tuple(results)


def gaussian_log_likelihood(x, mean, std, full=True, reduction="sum"):
    """
    Computes the log likelihood of a multivariate factorized Gaussian.

    The Gaussian log likelihood is
    `log p(x) = sum_i [- log_std_i - 0.5 * log (2 pi) - 0.5 (x_i - mu_i)^2 / exp(log_std_i)^2]`.
    """

    var = std**2
    eps = 1e-06

    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    log_likelihood  = -0.5 * (torch.log(var) + (mean - x)**2 / var)
    log_likelihood = torch.sum(log_likelihood, dim=tuple(range(1, len(log_likelihood.shape)))).unsqueeze(1)
    return log_likelihood

def vector_to_gaussian(x, min_std=0.0, fix_std=False):
    """
    Map network output to mean and stddev (via softplus) of Gaussian.

    [b, 2*d, ...] -> 2*[b, d, ...]
    """

    if fix_std:
        mu = x
        std = min_std * torch.ones_like(x)
    else:
        d = x.shape[1] // 2
        mu, std_param = x[:, :d], x[:, d:]
        std = F.softplus(std_param) + min_std

    return mu, std

class HeuristicInterventionEncoder(torch.nn.Module):
    """Intervention encoder"""

    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(-3.0))
        self.beta = torch.nn.Parameter(torch.tensor(15.0))
        self.gamma = torch.nn.Parameter(torch.tensor(15.0))

    def forward(self, inputs, eps=1.0e-4):
        """
        Given the means and standard deviations of the noise encoding encoder before and after
        interventions, computs the probabilities over intervention targets
        """
        dim_z = inputs.shape[1] // 2
        delta_epsilon = inputs[:, dim_z:]

        intervention_logits = (self.alpha + self.beta * torch.abs(delta_epsilon) + self.gamma * delta_epsilon**2)
        no_intervention_logit = torch.zeros((inputs.shape[0], 1), device=inputs.device)
        logits = torch.cat((no_intervention_logit, intervention_logits), dim=1)
        # noinspection PyUnresolvedReferences
        probs = torch.nn.functional.softmax(logits, dim=1)

        # To avoid potential sources of NaNs, we make sure that all probabilities are at least 2% or so
        probs = probs + eps
        probs = probs / torch.sum(probs, dim=1, keepdim=True)

        return probs

    def get_parameters(self):
        """Returns parameters for logging purposes"""
        return dict(alpha=self.alpha, beta=self.beta, gamma=self.gamma)

class ILCM(nn.Module):
    """
    Top-level class for generative models with
    - an SCM with a learned or fixed causal graph
    - separate encoder and decoder (i.e. a VAE) outputting noise encodings
    - VI over intervention targets
    """
    def __init__(
        self,
        causal_model,
        encoder,
        decoder,
        intervention_encoder,
        dim_z=8,
        intervention_prior=None,
        intervention_set="atomic_or_none",
        averaging_strategy="stochastic",
    ):
        super().__init__()
        self.dim_z = dim_z
        self.intervention_encoder = intervention_encoder
        self.averaging_strategy = averaging_strategy
        self.scm = causal_model

        intervention_prior = InterventionPrior(0, dim_z=dim_z, intervention_set=intervention_set)
        self.intervention_prior = intervention_prior
        self.n_interventions = self.intervention_prior.n_interventions
        self.register_buffer("_interventions", self.intervention_prior._masks.to(torch.float))

        self.encoder = encoder
        self.decoder = decoder

        self.register_buffer("dummy", torch.zeros([1]))  # Sole purpose is to track the device for self.sample()

    def sample(self, n, additional_noise=None, device=None):
        """
        Samples from the data-generating process in the weakly supervised setting.

        Returns:
        --------
        x1 : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Observed data point (before the intervention)
        x2 : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Observed data point (after a uniformly sampled intervention)
        intervention_labels : torch.Tensor of shape (batchsize,), dtype torch.int
            Obfuscated intervention label
        interventions : torch.Tensor of shape (batchsize, self.dim_z), dtype torch.bool
            Intervention masks
        """

        # Sample intervention
        interventions, intervention_labels = self.intervention_prior.sample(n)

        # Sample causal variables
        z1, z2 = self.scm.sample_weakly_supervised(n, interventions)

        # Push to data space
        x1, _ = self.decoder(z1)
        x2, _ = self.decoder(z2)

        # Optionally, add a small amount of observation noise to avoid numerical issues with
        # submanifolds
        if additional_noise:
            x1 += additional_noise * torch.randn(x1.size(), device=x1.device)
            x2 += additional_noise * torch.randn(x2.size(), device=x2.device)

        return x1, x2, z1, z2, intervention_labels, interventions

    def forward(
        self,
        x1,
        x2,
        interventions=None,
        beta=1.0,
        beta_intervention_target=None,
        pretrain_beta=None,
        full_likelihood=True,
        likelihood_reduction="sum",
        graph_mode="hard",
        graph_temperature=1.0,
        graph_samples=1,
        pretrain=False,
        model_interventions=True,
        deterministic_intervention_encoder=False,
        intervention_encoder_offset=1.0e-4,
        **kwargs,
    ):
        """
        Evaluates an observed data pair.

        Arguments:
        ----------
        x1 : torch.Tensor of shape (batchsize, DIM_X,), dtype torch.float
            Observed data point (before the intervention)
        x2 : torch.Tensor of shape (batchsize, DIM_X,), dtype torch.float
            Observed data point (after the intervention)
        interventions : None or torch.Tensor of shape (batchsize, DIM_Z,), dtype torch.float
            If not None, specifies the interventions

        Returns:
        --------
        log_prob : torch.Tensor of shape (batchsize, 1), dtype torch.float
            If `interventions` is not None: Conditional log likelihood
            `log p(x1, x2 | interventions)`.
            If `interventions` is None: Marginal log likelihood `log p(x1, x2)`.
        outputs : dict with str keys and torch.Tensor values
            Detailed breakdown of the model outputs and internals.
        """

        # Check inputs
        if beta_intervention_target is None:
            beta_intervention_target = beta
        if pretrain_beta is None:
            pretrain_beta = beta

        batchsize = x1.shape[0]
        feature_dims = list(range(1, len(x1.shape)))
        assert torch.all(torch.isfinite(x1)) and torch.all(torch.isfinite(x2))
        assert interventions is None

        # Pretraining
        if pretrain:
            return self.forward_pretrain(x1, x2, beta=pretrain_beta, full_likelihood=full_likelihood, likelihood_reduction=likelihood_reduction)

        # Get noise encoding means and stds
        e1_mean, e1_std = self.encoder.mean_std(x1)
        e2_mean, e2_std = self.encoder.mean_std(x2)

        # Compute intervention posterior
        intervention_posterior = self._encode_intervention(e1_mean, e2_mean, intervention_encoder_offset, deterministic_intervention_encoder)

        # Regularization terms
        e_norm, consistency_mse, _ = self._compute_latent_reg_consistency_mse(e1_mean, e1_std, e2_mean, e2_std, feature_dims, x1, x2, beta=beta)

        # Pretraining
        if pretrain:
            return self.forward_pretrain(x1, x2, beta=pretrain_beta, full_likelihood=full_likelihood, likelihood_reduction=likelihood_reduction)

        # Iterate over interventions
        log_posterior_eps, log_prior_eps = 0, 0
        log_posterior_int, log_prior_int, log_likelihood = 0, 0, 0
        mse, inverse_consistency_mse = 0, 0
        outputs = {}

        for (intervention, weight,) in self._iterate_interventions(intervention_posterior, deterministic_intervention_encoder, batchsize):
            # Sample from e1, e2 given intervention (including the projection to the counterfactual manifold)
            e1_proj, e2_proj, log_posterior1_proj, log_posterior2_proj = self._project_and_sample(e1_mean, e1_std, e2_mean, e2_std, intervention)

            # Compute ELBO terms
            (   log_likelihood_proj,
                log_posterior_eps_proj,
                log_posterior_int_proj,
                log_prior_eps_proj,
                log_prior_int_proj,
                mse_proj,
                inverse_consistency_mse_proj,
                outputs_,
            ) = self._compute_elbo_terms(
                x1,
                x2,
                e1_proj,
                e2_proj,
                feature_dims,
                full_likelihood,
                intervention,
                likelihood_reduction,
                log_posterior1_proj,
                log_posterior2_proj,
                weight,
                graph_mode=graph_mode,
                graph_samples=graph_samples,
                graph_temperature=graph_temperature,
                model_interventions=model_interventions,
            )

            # Sum up results
            log_posterior_eps += weight * log_posterior_eps_proj
            log_posterior_int += weight * log_posterior_int_proj
            log_prior_eps += weight * log_prior_eps_proj
            log_prior_int += weight * log_prior_int_proj
            log_likelihood += weight * log_likelihood_proj
            mse += weight * mse_proj
            inverse_consistency_mse += inverse_consistency_mse_proj

            # Some more bookkeeping
            for key, val in outputs_.items():
                val = val.unsqueeze(1)
                if key in outputs:
                    outputs[key] = torch.cat((outputs[key], val), dim=1)
                else:
                    outputs[key] = val

        loss = self._compute_outputs(
            beta,
            beta_intervention_target,
            consistency_mse,
            e1_std,
            e2_std,
            e_norm,
            intervention_posterior,
            log_likelihood,
            log_posterior_eps,
            log_posterior_int,
            log_prior_eps,
            log_prior_int,
            mse,
            outputs,
            inverse_consistency_mse,
        )

        return loss, outputs

    def _encode_intervention(self, e1_mean, e2_mean, intervention_encoder_offset, deterministic_intervention_encoder):
        intervention_encoder_inputs = torch.cat((e1_mean, e2_mean - e1_mean), dim=1)
        intervention_posterior = self.intervention_encoder(intervention_encoder_inputs, eps=intervention_encoder_offset)
        assert torch.all(torch.isfinite(intervention_posterior))

        # Deterministic intervention encoder: binarize, but use STE for gradients
        if deterministic_intervention_encoder:
            batchsize = e1_mean.shape[0]
            most_likely_intervention = torch.argmax(intervention_posterior, dim=1)  # (batchsize,)
            det_posterior = torch.zeros_like(intervention_posterior)
            det_posterior[torch.arange(batchsize), most_likely_intervention] = 1.0
            intervention_posterior = ( det_posterior.detach() + intervention_posterior - intervention_posterior.detach())

        return intervention_posterior

    def forward_pretrain(self, x1, x2, beta, full_likelihood=False, likelihood_reduction="sum"):
        assert torch.all(torch.isfinite(x1)) and torch.all(torch.isfinite(x2))
        feature_dims = list(range(1, len(x1.shape)))

        # Get noise encoding means and stds
        e1_mean, e1_std = self.encoder.mean_std(x1)
        e2_mean, e2_std = self.encoder.mean_std(x2)
        encoder_std = 0.5 * torch.mean(e1_std + e2_std, dim=1, keepdim=True)

        # Regularization terms
        e_norm, consistency_mse, beta_vae_loss = self._compute_latent_reg_consistency_mse(
            e1_mean,
            e1_std,
            e2_mean,
            e2_std,
            feature_dims,
            x1,
            x2,
            beta=beta,
            full_likelihood=full_likelihood,
            likelihood_reduction=likelihood_reduction,
        )

        # Pretraining loss
        outputs = dict(z_regularization=e_norm, consistency_mse=consistency_mse, encoder_std=encoder_std)

        return beta_vae_loss, outputs

    def log_likelihood(self, x1, x2, interventions=None, n_latent_samples=20, **kwargs):
        """
        Computes estimate of the log likelihood using importance weighting, like in IWAE.

        `log p(x) = log E_{inputs ~ q(inputs|x)} [ p(x|inputs) p(inputs) / q(inputs|x) ]`
        """

        # Copy each sample n_latent_samples times
        x1_ = self._expand(x1, repeats=n_latent_samples)
        x2_ = self._expand(x2, repeats=n_latent_samples)
        interventions_ = self._expand(interventions, repeats=n_latent_samples)

        # Evaluate ELBO
        negative_elbo, _ = self.forward(x1_, x2_, interventions_, beta=1.0)

        # Compute importance-weighted estimate of log likelihood
        log_likelihood = self._contract(-negative_elbo, mode="logmeanexp", repeats=n_latent_samples)

        return log_likelihood

    def encode_to_noise(self, x, deterministic=True):
        """
        Given data x, returns the noise encoding.

        Arguments:
        ----------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Data point to be encoded.
        deterministic : bool, optional
            If True, enforces deterministic encoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        e : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Noise encoding phi_e(x)
        """

        e, _ = self.encoder(x, deterministic=deterministic)
        return e

    def encode_to_causal(self, x, deterministic=True):
        """
        Given data x, returns the causal encoding.

        Arguments:
        ----------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Data point to be encoded.
        deterministic : bool, optional
            If True, enforces deterministic encoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        inputs : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Causal-variable encoding phi_z(x)
        """

        e, _ = self.encoder(x, deterministic=deterministic)
        adjacency_matrix = self._sample_adjacency_matrices(mode="deterministic", n=x.shape[0])
        z = self.scm.noise_to_causal(e, adjacency_matrix=adjacency_matrix)
        return z

    def decode_noise(self, e, deterministic=True):
        """
        Given noise encoding e, returns data x.

        Arguments:
        ----------
        e : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Noise-encoded data.
        deterministic : bool, optional
            If True, enforces deterministic decoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Decoded data point.
        """

        x, _ = self.decoder(e, deterministic=deterministic)
        return x

    def decode_causal(self, z, deterministic=True):
        """
        Given causal latents inputs, returns data x.

        Arguments:
        ----------
        inputs : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Causal latent variables.
        deterministic : bool, optional
            If True, enforces deterministic decoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Decoded data point.
        """

        adjacency_matrix = self._sample_adjacency_matrices(mode="deterministic", n=z.shape[0])
        e = self.scm.causal_to_noise(z, adjacency_matrix=adjacency_matrix)
        x, _ = self.decoder(e, deterministic=deterministic)
        return x

    def encode_decode(self, x, deterministic=True):
        """Auto-encode data and return reconstruction"""
        eps = self.encode_to_noise(x, deterministic=deterministic)
        x_reco = self.decode_noise(eps, deterministic=deterministic)
        return x_reco

    def encode_decode_pair(self, x1, x2, deterministic=True):
        """Auto-encode data pair and return latent representation and reconstructions"""

        # Get noise encoding means and stds
        e1_mean, e1_std = self.encoder.mean_std(x1)
        e2_mean, e2_std = self.encoder.mean_std(x2)

        # Compute intervention posterior
        intervention_encoder_inputs = torch.cat((e1_mean, e2_mean - e1_mean), dim=1)
        intervention_posterior = self.intervention_encoder(intervention_encoder_inputs)

        # Determine most likely intervention
        most_likely_intervention_idx = torch.argmax(intervention_posterior, dim=1).flatten()
        intervention = self._interventions[most_likely_intervention_idx]

        # Project to manifold
        e1_proj, e2_proj, log_posterior1_proj, log_posterior2_proj = self._project_and_sample(
            e1_mean, e1_std, e2_mean, e2_std, intervention, deterministic=deterministic
        )

        # Project back to data space
        x1_reco = self.decode_noise(e1_proj)
        x2_reco = self.decode_noise(e2_proj)

        return (x1_reco, x2_reco, e1_mean, e2_mean, e1_proj, e2_proj, intervention_posterior, most_likely_intervention_idx, intervention)

    def infer_intervention(self, x1, x2, deterministic=True):
        """Given data pair, infer intervention"""

        (x1_reco, x2_reco, e1_mean, e2_mean, e1_proj, e2_proj, intervention_posterior, most_likely_intervention_idx, intervention) = self.encode_decode_pair(x1, x2, deterministic=deterministic)
        return most_likely_intervention_idx, None, x2_reco

    def _iterate_interventions( self, intervention_posterior, deterministic_intervention_encoder, batchsize):
        if deterministic_intervention_encoder:
            most_likely_intervention = torch.argmax(intervention_posterior, dim=1)  # (batchsize,)
            interventions = self._interventions.unsqueeze(0).expand((batchsize, self._interventions.shape[0], self._interventions.shape[1]))
            intervention = interventions[torch.arange(batchsize), most_likely_intervention, :]
            weight = torch.ones((batchsize, 1), device=intervention_posterior.device)
            yield intervention, weight
        else:
            for intervention, weight in zip(self._interventions, intervention_posterior.T):
                intervention = intervention.unsqueeze(0).expand((batchsize, intervention.shape[0]))
                weight = weight.unsqueeze(1)  # (batchsize, 1)
                yield intervention, weight

    def _project_and_sample(self, e1_mean, e1_std, e2_mean, e2_std, intervention, deterministic=False):
        # Project to manifold
        (e1_mean_proj, e1_std_proj, e2_mean_proj, e2_std_proj) = self._project_to_manifold(intervention, e1_mean, e1_std, e2_mean, e2_std)

        # Sample noise
        e1_proj, log_posterior1_proj = gaussian_encode(e1_mean_proj, e1_std_proj)
        e2_proj, log_posterior2_proj = gaussian_encode(e2_mean_proj, e2_std_proj, reduction="none")

        # Sampling should preserve counterfactual consistency
        e2_proj = intervention * e2_proj + (1.0 - intervention) * e1_proj
        log_posterior2_proj = torch.sum(log_posterior2_proj * intervention, dim=1, keepdim=True)

        return e1_proj, e2_proj, log_posterior1_proj, log_posterior2_proj

    def _project_to_manifold(self, intervention, e1_mean, e1_std, e2_mean, e2_std):
        if self.averaging_strategy == "z2":
            lam = torch.ones_like(e1_mean)
        elif self.averaging_strategy in ["average", "mean"]:
            lam = 0.5 * torch.ones_like(e1_mean)
        elif self.averaging_strategy == "stochastic":
            lam = torch.rand_like(e1_mean)
        else:
            raise ValueError(f"Unknown averaging strategy {self.averaging_strategy}")

        projection_mean = lam * e1_mean + (1.0 - lam) * e2_mean
        projection_std = lam * e1_std + (1.0 - lam) * e2_std

        e1_mean = intervention * e1_mean + (1.0 - intervention) * projection_mean
        e1_std = intervention * e1_std + (1.0 - intervention) * projection_std
        e2_mean = intervention * e2_mean + (1.0 - intervention) * projection_mean
        e2_std = intervention * e2_std + (1.0 - intervention) * projection_std

        return e1_mean, e1_std, e2_mean, e2_std

    def _compute_latent_reg_consistency_mse(
        self,
        e1_mean,
        e1_std,
        e2_mean,
        e2_std,
        feature_dims,
        x1,
        x2,
        beta,
        full_likelihood=False,
        likelihood_reduction="sum",
    ):
        e1, log_posterior1 = gaussian_encode(e1_mean, e1_std)
        e2, log_posterior2 = gaussian_encode(e2_mean, e2_std)

        # Compute latent regularizer (useful early in training)
        e_norm = torch.sum(e1**2, 1, keepdim=True) + torch.sum(e2**2, 1, keepdim=True)

        # Compute consistency MSE
        consistency_x1_reco, log_likelihood1 = self.decoder(
            e1,
            eval_likelihood_at=x1,
            deterministic=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        consistency_x2_reco, log_likelihood2 = self.decoder(
            e2,
            eval_likelihood_at=x2,
            deterministic=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        consistency_mse = torch.sum((consistency_x1_reco - x1) ** 2, feature_dims).unsqueeze(1)
        consistency_mse += torch.sum((consistency_x2_reco - x2) ** 2, feature_dims).unsqueeze(1)

        # Compute prior and beta-VAE loss (for pre-training)
        log_prior1 = torch.sum(
            self.scm.base_density.log_prob(e1.reshape((-1, 1))).reshape((-1, self.dim_z)),
            dim=1,
            keepdim=True,
        )
        log_prior2 = torch.sum(
            self.scm.base_density.log_prob(e2.reshape((-1, 1))).reshape((-1, self.dim_z)),
            dim=1,
            keepdim=True,
        )
        beta_vae_loss = (-log_likelihood1 - log_likelihood2 + beta * (log_posterior1 + log_posterior2 - log_prior1 - log_prior2))

        return e_norm, consistency_mse, beta_vae_loss

    def _compute_outputs(
        self,
        beta,
        beta_intervention_target,
        consistency_mse,
        e1_std,
        e2_std,
        e_norm,
        intervention_posterior,
        log_likelihood,
        log_posterior_eps,
        log_posterior_int,
        log_prior_eps,
        log_prior_int,
        mse,
        outputs,
        inverse_consistency_mse,
    ):
        # Put together to compute the ELBO and beta-VAE loss
        kl_int = log_posterior_int - log_prior_int
        kl_eps = log_posterior_eps - log_prior_eps
        log_posterior = log_posterior_int + log_posterior_eps
        log_prior = log_prior_int + log_prior_eps
        kl = kl_eps + kl_int
        elbo = log_likelihood - kl
        beta_vae_loss = -log_likelihood + beta * kl_eps + beta_intervention_target * kl_int

        # Track individual components
        outputs["elbo"] = elbo
        outputs["beta_vae_loss"] = beta_vae_loss
        outputs["kl"] = kl
        outputs["kl_intervention_target"] = kl_int
        outputs["kl_e"] = kl_eps
        outputs["log_likelihood"] = log_likelihood
        outputs["log_posterior"] = log_posterior
        outputs["log_prior"] = log_prior
        outputs["intervention_posterior"] = intervention_posterior
        outputs["mse"] = mse
        outputs["consistency_mse"] = consistency_mse
        outputs["inverse_consistency_mse"] = inverse_consistency_mse
        outputs["z_regularization"] = e_norm
        outputs["encoder_std"] = 0.5 * torch.mean(e1_std + e2_std, dim=1, keepdim=True)

        return beta_vae_loss

    def _compute_elbo_terms(
        self,
        x1,
        x2,
        e1_proj,
        e2_proj,
        feature_dims,
        full_likelihood,
        intervention,
        likelihood_reduction,
        log_posterior1_proj,
        log_posterior2_proj,
        weight,
        model_interventions=True,
        **prior_kwargs,
    ):
        # Compute posterior q(e1, e2_I | I)
        log_posterior_eps_proj = log_posterior1_proj + log_posterior2_proj
        assert torch.all(torch.isfinite(log_posterior_eps_proj))

        # Compute posterior q(I)
        log_posterior_int_proj = weight * torch.log(weight)

        # Decode compute log likelihood / reconstruction error
        x1_reco_proj, log_likelihood1_proj, _ = self.decoder(
            e1_proj,
            eval_likelihood_at=x1,
            deterministic=True,
            return_std=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        x2_reco_proj, log_likelihood2_proj, _ = self.decoder(
            e2_proj,
            eval_likelihood_at=x2,
            deterministic=True,
            return_std=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        log_likelihood_proj = log_likelihood1_proj + log_likelihood2_proj
        assert torch.all(torch.isfinite(log_likelihood_proj))

        # Compute MSE
        mse_proj = torch.sum((x1_reco_proj - x1) ** 2, feature_dims).unsqueeze(1)
        mse_proj += torch.sum((x2_reco_proj - x2) ** 2, feature_dims).unsqueeze(1)

        # Compute inverse consistency MSE: |z - encode(decode(z))|^2
        e1_reencoded = self.encode_to_noise(x1_reco_proj, deterministic=False)
        e2_reencoded = self.encode_to_noise(x2_reco_proj, deterministic=False)
        inverse_consistency_mse_proj = torch.sum((e1_reencoded - e1_proj) ** 2, 1, keepdim=True)
        inverse_consistency_mse_proj += torch.sum((e2_reencoded - e2_proj) ** 2, 1, keepdim=True)

        # Compute prior p(e1, e2 | I)
        log_prior_eps_proj, outputs = self.scm.log_prob_noise_weakly_supervised(
            e1_proj,
            e2_proj,
            intervention,
            adjacency_matrix=None,
            include_intervened=model_interventions,
            include_nonintervened=False,
        )
        assert torch.all(torch.isfinite(log_prior_eps_proj))

        # Compute prior pi(I) = 1 / n_interventions
        log_prior_int_proj = -np.log(self.n_interventions) * torch.ones_like(log_prior_eps_proj)
        return (log_likelihood_proj,log_posterior_eps_proj, log_posterior_int_proj, log_prior_eps_proj, log_prior_int_proj, mse_proj, inverse_consistency_mse_proj, outputs)

    def load_state_dict(self, state_dict, strict=True):
        """Overloading the state dict loading so we can compute ancestor structure"""
        super().load_state_dict(state_dict, strict)
        self.scm._compute_ancestors()

    def _evaluate_prior(
        self,
        z1,
        z2,
        interventions,
        graph_mode="hard",
        graph_temperature=1.0,
        graph_samples=1,
        noise_centric=False,
        include_nonintervened=True,
    ):
        """
        Evaluates prior p(z1, z2) or p(epsilon1, epsilon2).

        If interventions is not None, explicitly marginalizes over all possible interventions with
        a uniform prior.
        """

        # Sample adjacency matrices
        z1 = self._expand(z1, repeats=graph_samples)
        z2 = self._expand(z2, repeats=graph_samples)
        adjacency_matrices = self._sample_adjacency_matrices(z1.shape[0], mode=graph_mode, temperature=graph_temperature)

        # If interventions are not specified, enumerate them
        if interventions is None:
            z1 = self._expand(z1)
            z2 = self._expand(z2)
            adjacency_matrices_ = self._expand(adjacency_matrices)
            interventions_ = self._enumerate_interventions(z1, z2)
        else:
            adjacency_matrices_ = adjacency_matrices
            interventions_ = interventions

        # Evaluate prior p(z1, z2|interventions)
        log_prob, outputs = self._evaluate_intervention_conditional_prior(
            z1,
            z2,
            interventions_,
            adjacency_matrices_,
            noise_centric=noise_centric,
            include_nonintervened=include_nonintervened,
        )

        # Marginalize over interventions
        if interventions is None:
            outputs = self._contract_dict(outputs)
            log_prob = self._contract(log_prob, mode="logmeanexp")  # Marginalize likelihood

        # Marginalize over adjacency matrices
        log_prob = self._contract(log_prob, repeats=graph_samples, mode="mean")
        outputs = self._contract_dict(outputs, repeats=graph_samples)

        return log_prob, outputs

    def _evaluate_intervention_conditional_prior(
        self,
        z1,
        z2,
        interventions,
        adjacency_matrices,
        noise_centric=False,
        include_nonintervened=True,
    ):
        """Evaluates conditional prior p(z1, z2|I)"""

        # Check inputs
        interventions = self._sanitize_interventions(interventions)

        # Evaluate conditional prior
        if noise_centric:
            log_prob, outputs = self.scm.log_prob_noise_weakly_supervised(
                z1, z2, interventions, adjacency_matrix=adjacency_matrices, include_nonintervened=include_nonintervened)
        else:
            log_prob, outputs = self.scm.log_prob_weakly_supervised(z1, z2, interventions, adjacency_matrix=adjacency_matrices)

        return log_prob, outputs

    def _sanitize_interventions(self, interventions):
        """Ensures correct dtype of interventions"""
        assert interventions.shape[1] == self.dim_z
        return interventions.to(torch.float)

    def _enumerate_interventions(self, z1, z2):
        """Generates interventions"""
        n = (z1.shape[0] // self._interventions.shape[0])  # z1 has shape (n_interventions * batchsize, DIM_Z) already
        return self._interventions.repeat_interleave(n, dim=0)

    def _expand(self, x, repeats=None):
        """
        Given x with shape (batchsize, components), repeats elements and returns a tensor of shape
        (batchsize * repeats, components)
        """

        if x is None:
            return None
        if repeats is None:
            repeats = len(self._interventions)

        unaffected_dims = tuple(1 for _ in range(1, len(x.shape)))
        x_expanded = x.repeat(repeats, *unaffected_dims)

        return x_expanded

    def _contract(self, x, mode="mean", repeats=None):
        """
        Given x with shape (batchsize * repeats, components), returns either
         - the mean over repeats, with shape (batchsize, components),
         - the logmeanexp over repeats, with shape (batchsize, components), or
         - the reshaped version with shape (batchsize, repeats, components).
        """

        assert mode in ["sum", "mean", "reshape", "logmeanexp"]

        if x is None:
            return None
        if len(x.shape) == 1:
            return self._contract(x.unsqueeze(1), mode, repeats)

        if repeats is None:
            repeats = len(self._interventions)

        # assert len(x.shape) == 2, x.shape
        y = x.reshape([repeats, -1] + list(x.shape[1:]))
        if mode == "sum":
            return torch.sum(y, dim=0)
        elif mode == "mean":
            return torch.mean(y, dim=0)
        elif mode == "logmeanexp":
            return logmeanexp(y, 0)
        elif mode == "reshape":
            return y.transpose(0, 1)
        else:
            raise ValueError(mode)

    def _contract_dict(self, data, repeats=None):
        """Given a dict of data, contracts each data variable approproately (see `_contract`)"""

        contracted_dict = {}
        for key, val in data.items():
            if key in MEAN_VARS:
                mode = "mean"
            elif key in LOGMEANEXP_VARS:
                mode = "logmeanexp"
            else:
                mode = "reshape"
            contracted_dict[key] = self._contract(val, mode, repeats)

        return contracted_dict

    def _sample_adjacency_matrices(self, *args, **kwargs):
        if self.scm.graph is None:
            return None
        return self.scm.graph.sample_adjacency_matrices(*args, **kwargs)

class InterventionPrior(nn.Module):
    """
    Prior categorical distribution over intervention targets, plus mapping to integer labels
    """

    def __init__(self, permutation=0, dim_z=2, intervention_set="atomic_or_none"):
        super().__init__()

        assert intervention_set in {"atomic_or_none", "atomic", "all"}
        self.intervention_set = intervention_set
        self.dim_z = dim_z

        _masks, self.n_interventions = self._generate_interventions()
        self.register_buffer("_masks", _masks)

        _permutation, _inverse_permutation = self._generate_permutation(permutation)
        self.register_buffer("_permutation", _permutation)
        self.register_buffer("_inverse_permutation", _inverse_permutation)

    def forward(self, intervention_label, convert_to_int=False):
        """Integer label to intervention mask"""

        assert len(intervention_label.shape) == 1
        assert torch.all(intervention_label >= 0)
        assert torch.all(intervention_label < self.n_interventions)

        # Blind label to default label
        intervention_idx = torch.index_select(self._permutation, 0, intervention_label)

        # Default label to binary mask
        intervention = torch.index_select(self._masks, 0, intervention_idx)

        # Covert to int if necessary
        if convert_to_int:
            intervention = intervention.to(torch.int)
        return intervention

    def inverse(self, intervention):
        """Intervention mask to integer label"""

        assert len(intervention.shape) == 2
        assert intervention.shape[1] == self.dim_z

        # Intervention mask to default label
        if self.intervention_set == "all":
            intervention_idx = 0
            for i, intervention_i in enumerate(intervention.T):
                intervention_idx = intervention_idx + 2**i * intervention_i
        else:  # Atomic interventions
            intervention_idx = 0
            for i, intervention_i in enumerate(intervention.T):
                intervention_idx = intervention_idx + intervention_i

        assert torch.all(intervention_idx < self.n_interventions)

        # Default label to blind label
        intervention_label = torch.index_select(self._inverse_permutation, 0, intervention_idx)
        return intervention_label

    def sample(self, n, convert_to_int=True):
        """Samples intervention targets from a uniform distribution"""
        intervention_labels = torch.randint(
            self.n_interventions, size=(n,), device=self._masks.device
        )
        interventions = self(intervention_labels, convert_to_int=convert_to_int)
        return interventions, intervention_labels

    def _generate_interventions(self):
        if self.intervention_set == "all":
            n_interventions = 2**self.dim_z
            masks = []
            for idx in range(n_interventions):
                masks.append(
                    torch.BoolTensor([(idx >> k) & 1 for k in range(0, self.dim_z)]).unsqueeze(0)
                )
        elif self.intervention_set == "atomic_or_none":
            n_interventions = self.dim_z + 1
            masks = [torch.BoolTensor([False for _ in range(self.dim_z)]).unsqueeze(0)]
            for idx in range(self.dim_z):
                masks.append(torch.BoolTensor([(idx == k) for k in range(self.dim_z)]).unsqueeze(0))
        elif self.intervention_set == "atomic":
            n_interventions = self.dim_z
            masks = []
            for idx in range(n_interventions):
                masks.append(torch.BoolTensor([(idx == k) for k in range(self.dim_z)]).unsqueeze(0))
        else:
            raise ValueError(f"Unknown intervention set {self.intervention_set}")

        assert len(masks) == n_interventions

        return torch.cat(masks, 0), n_interventions

    def _generate_permutation(self, permutation):
        """Helper function to generate a permutation matrix"""

        idx = list(range(self.n_interventions))

        # Find permutation
        permutation_ = None
        for i, perm in enumerate(itertools.permutations(idx)):
            if i == permutation:
                permutation_ = perm
                break

        assert permutation_ is not None

        permutation_tensor = torch.IntTensor(permutation_)
        inverse_permutation_tensor = torch.IntTensor(np.argsort(permutation_))

        return permutation_tensor, inverse_permutation_tensor