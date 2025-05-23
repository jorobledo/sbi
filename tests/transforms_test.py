# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import sys

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import (
    Exponential,
    LogNormal,
    MultivariateNormal,
    TransformedDistribution,
    Uniform,
)
from torch.distributions.transforms import (
    AffineTransform,
    ExpTransform,
    IndependentTransform,
    SigmoidTransform,
)

from sbi.utils import BoxUniform, MultipleIndependent, mcmc_transform, process_prior
from tests.user_input_checks_test import UserNumpyUniform


@pytest.mark.parametrize(
    "prior, target_transform",
    (
        (Uniform(-torch.ones(1), torch.ones(1)), SigmoidTransform),
        (BoxUniform(-torch.ones(2), torch.ones(2)), SigmoidTransform),
        (UserNumpyUniform(torch.zeros(2), torch.ones(2)), SigmoidTransform),
        (MultivariateNormal(torch.zeros(2), torch.eye(2)), AffineTransform),
        (LogNormal(loc=torch.zeros(1), scale=torch.ones(1)), ExpTransform),
    ),
)
def test_transforms(prior, target_transform):
    if isinstance(prior, UserNumpyUniform):
        prior, *_ = process_prior(
            prior,
            dict(lower_bound=torch.zeros(2), upper_bound=torch.ones(2)),
        )

    transform = mcmc_transform(prior)
    core_transform = transform._inv

    if isinstance(core_transform, IndependentTransform):
        core_transform = core_transform.base_transform

    if hasattr(core_transform, "parts"):
        transform_to_inspect = core_transform.parts[0]
    else:
        transform_to_inspect = core_transform

    assert isinstance(transform_to_inspect, target_transform)

    samples = prior.sample((2,))
    transformed_samples = transform(samples)
    assert torch.allclose(samples, transform.inv(transformed_samples))


@pytest.mark.parametrize(
    "prior, enable_transform",
    (
        (BoxUniform(zeros(5), ones(5)), True),
        (BoxUniform(zeros(1), ones(1)), True),
        (BoxUniform(zeros(5), ones(5)), False),
        (MultivariateNormal(zeros(5), eye(5)), True),
        (Exponential(rate=ones(1)), True),
        (LogNormal(zeros(1), ones(1)), True),
        (
            MultipleIndependent([
                Exponential(rate=ones(1)),
                BoxUniform(zeros(5), ones(5)),
            ]),
            True,
        ),
    ),
)
def test_mcmc_transform(prior, enable_transform):
    """
    Test whether the transform for MCMC returns the log_abs_det in the correct shape.
    """

    num_samples = 1000
    prior, _, _ = process_prior(prior)
    tf = mcmc_transform(prior, enable_transform=enable_transform)

    samples = prior.sample((num_samples,))
    unconstrained_samples = tf(samples)
    samples_original = tf.inv(unconstrained_samples)

    log_abs_det = tf.log_abs_det_jacobian(samples_original, unconstrained_samples)
    assert log_abs_det.shape == torch.Size([num_samples])


@pytest.mark.xfail(
    reason="Known issue in PyTorch https://github.com/pytorch/pytorch/issues/20682",
    strict=True,
    raises=AssertionError,
)
def test_transformed_dist_support():
    """
    Test whether the support of the transformed distribution is correct.
    This is a known pytorch issue
    SBI issue: https://github.com/sbi-dev/sbi/issues/738
    PyTorch issue: https://github.com/pytorch/pytorch/issues/20682
    """
    base = Uniform(zeros(1), ones(1))
    dist = TransformedDistribution(base, AffineTransform(zeros(1), ones(1)))
    assert not dist.support.check(100 * ones(1)).item(), (
        "Support check failed with python version={} and torch version={}".format(
            sys.version, torch.__version__
        )
    )
