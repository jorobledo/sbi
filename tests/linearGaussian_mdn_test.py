# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import Tensor, eye, ones, zeros
from torch.distributions import MultivariateNormal, Uniform

from sbi.inference import (
    NLE,
    NPE,
    DirectPosterior,
    MCMCPosterior,
    likelihood_estimator_based_potential,
)
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.metrics import check_c2st


@pytest.mark.parametrize(
    "method", (NPE, pytest.param(NLE, marks=[pytest.mark.slow, pytest.mark.mcmc]))
)
def test_mdn_inference_with_different_methods(method, mcmc_params_accurate: dict):
    num_dim = 2
    x_o = torch.tensor([[1.0, 0.0]])
    num_samples = 500
    num_simulations = 2000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    def simulator(theta: Tensor) -> Tensor:
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = method(density_estimator="mdn")

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    estimator = inference.append_simulations(theta, x).train()
    if method == NPE:
        posterior = DirectPosterior(posterior_estimator=estimator, prior=prior)
    else:
        potential_fn, theta_transform = likelihood_estimator_based_potential(
            likelihood_estimator=estimator, prior=prior, x_o=x_o
        )
        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            method="slice_np_vectorized",
            **mcmc_params_accurate,
        )

    samples = posterior.sample((num_samples,), x=x_o)

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"{method.__name__}")


def test_mdn_with_1D_uniform_prior():
    """
    Note, we have this test because for 1D uniform priors, mdn log prob evaluation
    results in batch_size x batch_size return. This is probably because Uniform does
    not allow for event dimension > 1 and somewhere in pyknos it is used as if this was
    possible.
    Casting to BoxUniform solves it.
    """
    num_dim = 1
    x_o = torch.tensor([[1.0]])
    num_simulations = 100
    num_samples = 100

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior = Uniform(low=torch.zeros(num_dim), high=torch.ones(num_dim))

    def simulator(theta: Tensor) -> Tensor:
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = NPE(density_estimator="mdn")

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    posterior_estimator = inference.append_simulations(theta, x).train()
    posterior = DirectPosterior(posterior_estimator=posterior_estimator, prior=prior)
    samples = posterior.sample((num_samples,), x=x_o)
    log_probs = posterior.log_prob(samples, x=x_o)

    assert log_probs.shape == torch.Size([num_samples])
