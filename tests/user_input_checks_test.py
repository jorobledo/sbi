# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pytest
import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN
from torch import Tensor, eye, nn, ones, zeros
from torch.distributions import (
    Beta,
    Distribution,
    Exponential,
    Gamma,
    MultivariateNormal,
    Uniform,
)

from sbi.inference import NPE_A, NPE_C, simulate_for_sbi
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.simulators import linear_gaussian
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.utils import mcmc_transform, within_support
from sbi.utils.torchutils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
    process_x,
)
from sbi.utils.user_input_checks_utils import (
    CustomPriorWrapper,
    MultipleIndependent,
    OneDimPriorWrapper,
    PytorchReturnTypeWrapper,
)


class UserNumpyUniform:
    """User defined numpy uniform prior.

    Used for testing to mimick a user-defined prior with valid .sample and .log_prob
    methods.
    """

    def __init__(self, lower: Tensor, upper: Tensor, return_numpy: bool = False):
        self.lower = lower
        self.upper = upper
        self.dist = BoxUniform(lower, upper)
        self.return_numpy = return_numpy

    def sample(self, sample_shape=torch.Size([])):
        samples = self.dist.sample(sample_shape)
        return samples.numpy() if self.return_numpy else samples

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = self.dist.log_prob(values)
        return log_probs.numpy() if self.return_numpy else log_probs


def linear_gaussian_no_batch(theta):
    """Identity simulator throwing assertion error when called on a batch."""
    assert theta.ndim == 1, "cant handle batches."
    return MultivariateNormal(theta, eye(theta.numel())).sample()


def numpy_linear_gaussian(theta):
    """Linear Gaussian simulator wrapped to get and return numpy."""
    return diagonal_linear_gaussian(torch.as_tensor(theta, dtype=torch.float32)).numpy()


def list_simulator(theta):
    return list(theta)


def matrix_simulator(theta):
    """Return a 2-by-2 matrix."""
    assert theta.numel() == 4
    return theta.reshape(1, 2, 2)


# Set default tensor locally to reach tensors in fixtures.
torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize(
    "wrapper, prior, kwargs",
    (
        (
            CustomPriorWrapper,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
            dict(lower_bound=zeros(3), upper_bound=ones(3)),
        ),
        (
            PytorchReturnTypeWrapper,
            BoxUniform(zeros(3, dtype=torch.float64), ones(3, dtype=torch.float64)),
            dict(),
        ),
        (
            OneDimPriorWrapper,
            Exponential(torch.tensor([3.0])),
            dict(),
        ),
    ),
)
def test_prior_wrappers(wrapper, prior, kwargs):
    """Test prior wrappers to pytorch distributions."""
    prior = wrapper(prior, **kwargs)

    # use 2 here to test for minimal case >1
    batch_size = 2
    theta = prior.sample((batch_size,))
    assert isinstance(theta, Tensor)
    assert theta.shape[0] == batch_size

    # Test log prob on batch of thetas.
    log_probs = prior.log_prob(theta)
    assert isinstance(log_probs, Tensor)
    assert log_probs.shape[0] == batch_size

    # Test return type
    assert prior.sample().dtype == torch.float32

    # Test support check.
    within_support(prior, prior.sample((2,)))
    # Test transform
    mcmc_transform(prior)

    # For 1D priors, the `log_prob()` should not have a batch dim.
    assert len(prior.log_prob(prior.sample((10,))).shape) == 1


def test_reinterpreted_batch_dim_prior():
    """Test whether the right warning and error are raised for reinterpreted priors."""

    # Both must raise ValueError because we don't reinterpret batch dims anymore.
    with pytest.raises(ValueError):
        process_prior(Uniform(zeros(3), ones(3)))
    with pytest.raises(ValueError):
        process_prior(MultivariateNormal(zeros(2, 3), ones(3)))

    # This must pass without warnings or errors.
    process_prior(BoxUniform(zeros(3), ones(3)))


@pytest.mark.parametrize(
    "prior",
    (
        pytest.param(Uniform(0.0, 1.0), marks=pytest.mark.xfail),  # scalar prior.
        pytest.param(
            Uniform(zeros((1, 3)), ones((1, 3))), marks=pytest.mark.xfail
        ),  # batch shape > 1.
        pytest.param(
            MultivariateNormal(zeros(3, 3), eye(3)),
            marks=pytest.mark.xfail,
        ),  # batch shape > 1.
        pytest.param(
            Uniform(zeros(3), ones(3)), marks=pytest.mark.xfail
        ),  # batched uniform.
        Uniform(zeros(1), ones(1)),
        BoxUniform(zeros(3), ones(3)),
        MultivariateNormal(zeros(3), eye(3)),
        UserNumpyUniform(zeros(3), ones(3), return_numpy=False),
        UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        BoxUniform(zeros(3, dtype=torch.float64), ones(3, dtype=torch.float64)),
        [
            Uniform(zeros(1), ones(1)),
            Uniform(zeros(1), ones(1)),
        ],  # multiple independent prior
        pytest.param(
            [
                UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
                UserNumpyUniform(zeros(3), ones(3), return_numpy=False),
                Uniform(zeros(1), ones(1)),
            ],  # combination of multiple independent custom priors
            marks=pytest.mark.xfail(
                reason="custom priors cannot be used in MultipleIndependent priors "
                "(unless they support .to)."
            ),
        ),
    ),
)
def test_process_prior(prior):
    prior, parameter_dim, numpy_simulator = process_prior(
        prior,
        custom_prior_wrapper_kwargs=dict(lower_bound=zeros(3), upper_bound=ones(3)),
    )

    batch_size = 2
    theta = prior.sample((batch_size,))
    assert theta.shape == torch.Size((
        batch_size,
        parameter_dim,
    )), "Number of sampled parameters must match batch size."
    assert prior.log_prob(theta).shape[0] == batch_size, (
        "Number of log probs must match number of input values."
    )


@pytest.mark.parametrize(
    "x, x_shape",
    (
        (ones(3), torch.Size([3])),
        (ones(1, 3), torch.Size([3])),
        (ones(10, 3), torch.Size([10, 3])),  # 2D data / iid NPE
        pytest.param(ones(10, 3), None),  # 2D data / iid NPE without x_shape
        (ones(10, 10), torch.Size([10])),  # iid likelihood based
    ),
)
def test_process_x(x, x_shape):
    process_x(x, x_shape)


@pytest.mark.parametrize(
    "simulator, prior, x_shape",
    (
        (diagonal_linear_gaussian, BoxUniform(zeros(1), ones(1)), (1,)),
        (diagonal_linear_gaussian, BoxUniform(zeros(2), ones(2)), (2,)),
        (numpy_linear_gaussian, UserNumpyUniform(zeros(2), ones(2), True), (2,)),
        (linear_gaussian_no_batch, BoxUniform(zeros(2), ones(2)), (2,)),
        pytest.param(
            list_simulator,
            BoxUniform(zeros(2), ones(2)),
            (2,),
        ),
        # test simulator with 2D data.
        (lambda _: torch.randn(10, 2), BoxUniform(zeros(2), ones(2)), (10, 2)),
    ),
)
def test_process_simulator(simulator: Callable, prior: Distribution, x_shape: Tuple):
    prior, theta_dim, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    n_batch = 1
    x = simulator(prior.sample((n_batch,)))

    assert isinstance(x, Tensor), "Processed simulator must return Tensor."
    assert x.shape[0] == n_batch, (
        "Processed simulator must return as many data points as parameters in batch."
    )
    assert x.shape[1:] == x_shape


@pytest.mark.parametrize(
    "simulator, prior",
    (
        (
            linear_gaussian_no_batch,
            BoxUniform(zeros(3), ones(3)),
        ),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        ),
        (diagonal_linear_gaussian, BoxUniform(zeros(3), ones(3))),
        (
            diagonal_linear_gaussian,
            BoxUniform(zeros(3, dtype=torch.float64), ones(3, dtype=torch.float64)),
        ),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        ),
        (
            diagonal_linear_gaussian,
            [
                Gamma(ones(1), ones(1)),
                Beta(ones(1), ones(1)),
                MultivariateNormal(zeros(2), eye(2)),
            ],
        ),
        pytest.param(
            list_simulator,
            BoxUniform(zeros(3), ones(3)),
        ),
    ),
)
def test_prepare_sbi_problem(simulator: Callable, prior):
    """Test user interface by passing different kinds of simulators, prior and data.

    Args:
        simulator: simulator function
        prior: prior as defined by the user (pytorch, scipy, custom)
        x_shape: shape of data as defined by the user.
    """
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)

    # check batch sims and type
    n_batch = 5
    assert simulator(prior.sample((n_batch,))).shape[0] == n_batch
    assert isinstance(simulator(prior.sample((1,))), Tensor)
    assert prior.sample().dtype == torch.float32


@pytest.mark.parametrize("snpe_method", [NPE_A, NPE_C])
@pytest.mark.parametrize(
    "user_simulator, user_prior",
    (
        (
            diagonal_linear_gaussian,
            BoxUniform(zeros(3, dtype=torch.float64), ones(3, dtype=torch.float64)),
        ),
        (linear_gaussian_no_batch, BoxUniform(zeros(3), ones(3))),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        ),
        (diagonal_linear_gaussian, BoxUniform(zeros(3), ones(3))),
        (linear_gaussian_no_batch, BoxUniform(zeros(3), ones(3))),
        (list_simulator, BoxUniform(-ones(3), ones(3))),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        ),
        (
            diagonal_linear_gaussian,
            (
                Gamma(ones(1), ones(1)),
                Beta(ones(1), ones(1)),
                MultivariateNormal(zeros(2), eye(2)),
            ),
        ),
        (diagonal_linear_gaussian, Exponential(torch.tensor([3.0]))),
    ),
)
def test_inference_with_user_sbi_problems(
    snpe_method: type, user_simulator: Callable, user_prior
):
    """
    Test inference with combinations of user defined simulators, priors and x_os.
    """

    prior, _, prior_returns_numpy = process_prior(user_prior)
    simulator = process_simulator(user_simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)
    inference = snpe_method(
        prior=prior,
        density_estimator="mdn_snpe_a" if snpe_method == NPE_A else "maf",
        show_progress_bars=False,
    )

    # Run inference.
    theta, x = simulate_for_sbi(simulator, prior, 100)
    x_o = torch.zeros(x.shape[1])
    posterior_estimator = inference.append_simulations(theta, x).train(max_num_epochs=2)

    # Build posterior.
    if snpe_method == NPE_A:
        if not isinstance(prior, (MultivariateNormal, BoxUniform, DirectPosterior)):
            with pytest.raises(AssertionError):
                # NPE-A does not support priors yet.
                posterior_estimator = inference.correct_for_proposal()
                _ = DirectPosterior(
                    posterior_estimator=posterior_estimator, prior=prior
                ).set_default_x(x_o)
        else:
            _ = DirectPosterior(
                posterior_estimator=posterior_estimator, prior=prior
            ).set_default_x(x_o)
    else:
        _ = DirectPosterior(
            posterior_estimator=posterior_estimator, prior=prior
        ).set_default_x(x_o)


@pytest.mark.parametrize(
    "dists",
    [
        pytest.param(
            [Beta(ones(1), 2 * ones(1))], marks=pytest.mark.xfail
        ),  # single dist.
        pytest.param(
            [Gamma(ones(2), ones(1))], marks=pytest.mark.xfail
        ),  # single batched dist.
        pytest.param(
            [
                Gamma(ones(2), ones(1)),
                MultipleIndependent([
                    Uniform(zeros(1), ones(1)),
                    Uniform(zeros(1), ones(1)),
                ]),
            ],
            marks=pytest.mark.xfail,
        ),  # nested definition.
        pytest.param(
            [Uniform(0, 1), Beta(1, 2)], marks=pytest.mark.xfail
        ),  # scalar dists.
        [Uniform(zeros(1), ones(1)), Uniform(zeros(1), ones(1))],
        [
            Gamma(ones(1), ones(1)),
            Uniform(zeros(1), ones(1)),
            Beta(ones(1), 2 * ones(1)),
        ],
        [MultivariateNormal(zeros(3), eye(3)), Gamma(ones(1), ones(1))],
    ],
)
def test_independent_joint_shapes_and_samples(dists):
    """Test return shapes and validity of samples by comparing to samples generated
    from underlying distributions individually."""

    # Fix the seed for reseeding within this test.
    seed = 0

    joint = MultipleIndependent(dists)

    # Check shape of single sample and log prob
    sample = joint.sample()
    assert sample.shape == torch.Size([joint.ndims])
    assert joint.log_prob(sample).shape == torch.Size([1])

    num_samples = 10

    # seed sampling for later comparison.
    torch.manual_seed(seed)
    samples = joint.sample((num_samples,))
    log_probs = joint.log_prob(samples)

    # Check sample and log_prob return shapes.
    assert samples.shape == torch.Size([
        num_samples,
        joint.ndims,
    ]) or samples.shape == torch.Size([num_samples])
    assert log_probs.shape == torch.Size([num_samples])

    # Seed again to get same samples by hand.
    torch.manual_seed(seed)
    true_samples = []
    true_log_probs = []

    # Get samples and log probs by hand.
    for d in dists:
        sample = d.sample((num_samples,))
        true_samples.append(sample)
        true_log_probs.append(d.log_prob(sample).reshape(num_samples, -1))

    # collect in Tensor.
    true_samples = torch.cat(true_samples, dim=-1)
    true_log_probs = torch.cat(true_log_probs, dim=-1).sum(-1)

    # Check whether independent joint sample equal individual samples.
    assert (true_samples == samples).all()
    assert (true_log_probs == log_probs).all()

    # Check support attribute.
    within_support = joint.support.check(true_samples)
    assert within_support.all()


def test_invalid_inputs():
    dists = [
        Gamma(ones(1), ones(1)),
        Uniform(zeros(1), ones(1)),
        Beta(ones(1), 2 * ones(1)),
    ]
    joint = MultipleIndependent(dists)

    # Test too many input dimensions.
    with pytest.raises(AssertionError):
        joint.log_prob(ones(10, 4))

    # Test nested construction.
    with pytest.raises(AssertionError):
        MultipleIndependent([joint])

    # Test 3D value.
    with pytest.raises(AssertionError):
        joint.log_prob(ones(10, 4, 1))


@pytest.mark.parametrize(
    "arg",
    [
        ("maf"),
        ("MAF"),
        pytest.param(
            "nn",
            marks=pytest.mark.xfail(
                AssertionError,
                reason=(
                    "custom density estimator must be a function return nn.Module, "
                    "not the nn.Module."
                ),
            ),
        ),
        ("fun"),
    ],
)
def test_passing_custom_density_estimator(arg):
    x_numel = 2
    y_numel = 2
    hidden_features = 10
    num_components = 1
    mdn = MultivariateGaussianMDN(
        features=x_numel,
        context_features=y_numel,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(y_numel, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )
    if arg == "nn":
        # Just the nn.Module.
        density_estimator = mdn
    elif arg == "fun":
        # A function returning the nn.Module.
        density_estimator = lambda batch_theta, batch_x: mdn
    else:
        density_estimator = arg
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    _ = NPE_C(prior=prior, density_estimator=density_estimator)


@pytest.mark.parametrize(
    "num_simulations, simulation_batch_size, num_workers, \
        use_process_simulator",
    [
        (0, None, 1, True),
        (10, None, 1, True),
        (100, 10, 1, True),
        (100, None, 2, True),
        (1000, 50, 4, True),
        (100, 10, 2, False),
    ],
)
def test_simulate_for_sbi(
    num_simulations, simulation_batch_size, num_workers, use_process_simulator
):
    """Test the simulate_for_sbi function with various configurations."""
    num_dim = 3
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior = BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def failing_simulator(theta):
        if isinstance(theta, np.ndarray):
            raise TypeError("This simulator does not support numpy arrays.")
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    simulator = (
        failing_simulator
        if not use_process_simulator
        else process_simulator(failing_simulator, prior, False)
    )

    if num_simulations == 0:
        theta, x = simulate_for_sbi(
            simulator=simulator,
            proposal=prior,
            num_simulations=num_simulations,
            simulation_batch_size=simulation_batch_size,
            num_workers=num_workers,
        )
        assert theta.numel() == 0, (
            "Theta should be an empty tensor when num_simulations=0"
        )
        assert x.numel() == 0, "x should be an empty tensor when num_simulations=0"
    else:
        if not use_process_simulator and num_workers > 1:
            with pytest.raises(TypeError, match="For multiprocessing"):
                simulate_for_sbi(
                    simulator=simulator,
                    proposal=prior,
                    num_simulations=num_simulations,
                    simulation_batch_size=simulation_batch_size,
                    num_workers=num_workers,
                )
        else:
            theta, x = simulate_for_sbi(
                simulator=simulator,
                proposal=prior,
                num_simulations=num_simulations,
                simulation_batch_size=simulation_batch_size,
                num_workers=num_workers,
            )
            assert theta.shape[0] == num_simulations, (
                "Theta should have num_simulations rows"
            )
            assert x.shape[0] == num_simulations, "x should have num_simulations rows"
            assert theta.shape[1] == num_dim, "Theta should have num_dim columns"
            assert x.shape[1] == num_dim, "x should have num_dim columns"
