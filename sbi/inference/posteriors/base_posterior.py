# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import abstractmethod
from typing import Any, Dict, Optional, Union
from warnings import warn

import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor

from sbi.inference.potentials.base_potential import (
    BasePotential,
    CustomPotential,
    CustomPotentialWrapper,
)
from sbi.sbi_types import Array, Shape, TorchTransform
from sbi.utils.sbiutils import gradient_ascent
from sbi.utils.torchutils import ensure_theta_batched, process_device
from sbi.utils.user_input_checks import process_x


class NeuralPosterior:
    r"""Posterior :math:`p(\theta|x)` with `log_prob()` and `sample()` methods.

    All inference methods in sbi train a neural network which is then used to obtain
    the posterior distribution. The `NeuralPosterior` class wraps the trained network
    such that one can directly evaluate the (unnormalized) log probability and draw
    samples from the posterior.
    """

    def __init__(
        self,
        potential_fn: Union[BasePotential, CustomPotential],
        theta_transform: Optional[TorchTransform] = None,
        device: Optional[Union[str, torch.device]] = None,
        x_shape: Optional[torch.Size] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples. Must be a
                `BasePotential` or a `Callable` which takes `theta` and `x_o` as inputs.
            theta_transform: Transformation that will be applied during sampling.
                Allows to perform, e.g. MCMC in unconstrained space.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Deprecated, should not be passed.
        """
        if x_shape is not None:
            warn(
                "x_shape is not None. However, passing x_shape to the `Posterior` is "
                "deprecated and will be removed in a future release of `sbi`.",
                stacklevel=2,
            )

        # Wrap custom potential functions to adhere to the `BasePotential` interface.
        if not isinstance(potential_fn, BasePotential):
            potential_device = "cpu" if device is None else device
            potential_fn = CustomPotentialWrapper(
                potential_fn, prior=None, x_o=None, device=potential_device
            )

        self._device = process_device(potential_fn.device if device is None else device)

        self.potential_fn = potential_fn

        if theta_transform is None:
            self.theta_transform = torch_tf.IndependentTransform(
                torch_tf.identity_transform, reinterpreted_batch_ndims=1
            )
        else:
            self.theta_transform = theta_transform

        self._map = None
        self._purpose = ""

        # If the sampler interface (#573) is used, the user might have passed `x_o`
        # already to the potential function builder. If so, this `x_o` will be used
        # as default x.
        self._x = self.potential_fn.return_x_o()

    def potential(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""Evaluates $\theta$ under the potential that is used to sample the posterior.

        The potential is the unnormalized log-probability of $\theta$ under the
        posterior.

        Args:
            theta: Parameters $\theta$.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
        """
        self.potential_fn.set_x(self._x_else_default_x(x))

        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device), track_gradients=track_gradients
        )

    @abstractmethod
    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """See child classes for docstring."""
        pass

    @abstractmethod
    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        max_sampling_batch_size: int = 10_000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        """See child classes for docstring."""
        pass

    @property
    def default_x(self) -> Optional[Tensor]:
        """Return default x used by `.sample(), .log_prob` as conditioning context."""
        return self._x

    @default_x.setter
    def default_x(self, x: Tensor) -> None:
        """See `set_default_x`."""
        self.set_default_x(x)

    def set_default_x(self, x: Tensor) -> "NeuralPosterior":
        """Set new default x for `.sample(), .log_prob` to use as conditioning context.

        Reset the MAP stored for the old default x if applicable.

        This is a pure convenience to avoid having to repeatedly specify `x` in calls to
        `.sample()` and `.log_prob()` - only $\theta$ needs to be passed.

        This convenience is particularly useful when the posterior is focused, i.e.
        has been trained over multiple rounds to be accurate in the vicinity of a
        particular `x=x_o` (you can check if your posterior object is focused by
        printing it).

        NOTE: this method is chainable, i.e. will return the NeuralPosterior object so
        that calls like `posterior.set_default_x(my_x).sample(mytheta)` are possible.

        Args:
            x: The default observation to set for the posterior $p(\theta|x)$.
        Returns:
            `NeuralPosterior` that will use a default `x` when not explicitly passed.
        """
        self._x = process_x(x, x_event_shape=None).to(self._device)
        self._map = None
        return self

    def _x_else_default_x(self, x: Optional[Array]) -> Tensor:
        if x is not None:
            # New x, reset posterior sampler.
            self._posterior_sampler = None
            return process_x(x, x_event_shape=None)
        elif self.default_x is None:
            raise ValueError(
                "Context `x` needed when a default has not been set."
                "If you'd like to have a default, use the `.set_default_x()` method."
            )
        else:
            return self.default_x

    def _calculate_map(
        self,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """Calculates the maximum-a-posteriori estimate (MAP).
        See `map()` method of child classes for docstring.
        """

        if init_method == "posterior":
            inits = self.sample((num_init_samples,))
        elif init_method == "proposal":
            inits = self.proposal.sample((num_init_samples,))  # type: ignore
        elif isinstance(init_method, Tensor):
            inits = init_method
        else:
            raise ValueError

        return gradient_ascent(
            potential_fn=self.potential_fn,
            inits=inits,
            theta_transform=self.theta_transform,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
        )[0]

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The MAP is obtained by running gradient
        ascent from a given number of starting positions (samples from the posterior
        with the highest log-probability). After the optimization is done, we select the
        parameter set that has the highest log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
            x: Deprecated - use `.set_default_x()` prior to `.map()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a
                tensor, the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                for score-based estimators (thus, the default is `1000`.)
            show_progress_bars: Whether to show a progressbar during sampling from
                the posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.
        Returns:
            The MAP estimate.
        """
        if x is not None:
            raise ValueError(
                "Passing `x` directly to `.map()` has been deprecated."
                "Use `.self_default_x()` to set `x`, and then run `.map()` "
            )

        if self.default_x is None:
            raise ValueError(
                "Default `x` has not been set."
                "To set the default, use the `.set_default_x()` method."
            )

        if self._map is None or force_update:
            self.potential_fn.set_x(self.default_x)
            self._map = self._calculate_map(
                num_iter=num_iter,
                num_to_optimize=num_to_optimize,
                learning_rate=learning_rate,
                init_method=init_method,
                num_init_samples=num_init_samples,
                save_best_every=save_best_every,
                show_progress_bars=show_progress_bars,
            )
        return self._map

    def __repr__(self):
        desc = f"""{self.__class__.__name__} sampler for potential_fn=<{
            self.potential_fn.__class__.__name__
        }>"""
        return desc

    def __str__(self):
        desc = f"Posterior p(θ|x) of type {self.__class__.__name__}. {self._purpose}"
        return desc

    def __getstate__(self) -> Dict:
        """Returns the state of the object that is supposed to be pickled.

        Returns:
            Dictionary containing the state.
        """
        return self.__dict__

    def __setstate__(self, state_dict: Dict):
        """Sets the state when being loaded from pickle.

        For developers: for any new attribute added to `NeuralPosterior`, we have to
        add an entry here using `check_warn_and_setstate()`.

        Args:
            state_dict: State to be restored.
        """
        self.__dict__ = state_dict
