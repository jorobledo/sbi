import torch
from sbi.utils.torchutils import BoxUniform
import time
import pytest

def sample_and_time(prior, device, N=100000000):
    prior.to(device)
    start_time = time.time()
    prior.sample((N,))
    end_time = time.time()
    return prior, end_time - start_time

def sample_and_time_(prior, device, N=100000000):
    prior = prior._to(device)
    start_time = time.time()
    prior.sample((N,))
    end_time = time.time()
    return prior, end_time - start_time

@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_sampling_time(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this machine.")

    prior = BoxUniform(torch.tensor([0, 1]), torch.tensor([1, 2]), device="cpu")
    assert prior.low.device.type == "cpu"

    prior, sampling_time = sample_and_time(prior, device)
    print(f"Sampling time on {device}: {sampling_time:.6f} seconds")
    assert sampling_time > 0  # Ensure that timing is measured

    assert prior.low.device.type == device

    prior, sampling_time = sample_and_time_(prior, device)
    print(f"Sampling time on {device}: {sampling_time:.6f} seconds")
    assert sampling_time > 0  # Ensure that timing is measured

    assert prior.low.device.type == device
