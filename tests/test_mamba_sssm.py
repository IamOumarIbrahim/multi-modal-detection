"""Tests for simplified PyTorch Mamba-SSSM module."""

import pytest
import torch
from mmsar.postprocessing.base import PostProcessor
from mmsar.postprocessing.mamba_sssm import MambaSSSMModel, MambaSSSMPostProcessor


def test_mamba_sssm_output_shape() -> None:
    # (a) Batch of synthetic confidence sequences produces logits of expected shape
    batch_size = 4
    seq_len = 25
    d_model = 16
    d_state = 8

    model = MambaSSSMModel(in_features=1, d_model=d_model, d_state=d_state)
    x = torch.rand(batch_size, seq_len, 1)
    logits = model(x)

    assert logits.shape == (batch_size, seq_len), (
        f"Expected logits shape {(batch_size, seq_len)}, got {logits.shape}"
    )


def test_mamba_sssm_training_loss_decrease(capsys: pytest.CaptureFixture[str]) -> None:
    # (b) Train on a small synthetic dataset with a known threshold-crossing rule
    # for a fixed number of steps and assert final loss < initial loss, with values printed
    torch.manual_seed(42)

    # Synthetic dataset: confidence sequence where values >= 0.6 have label 1, else 0
    sequences = [
        [0.1, 0.2, 0.7, 0.8, 0.1, 0.9, 0.2, 0.8],
        [0.8, 0.9, 0.1, 0.2, 0.7, 0.8, 0.2, 0.1],
        [0.1, 0.1, 0.2, 0.2, 0.8, 0.9, 0.7, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.1, 0.2],
    ]
    labels = [[1 if val >= 0.6 else 0 for val in seq] for seq in sequences]

    processor = MambaSSSMPostProcessor(d_model=16, d_state=8, device="cpu")
    assert isinstance(processor, PostProcessor)

    epochs = 40
    losses = processor.fit(sequences, labels, epochs=epochs, lr=0.03)

    initial_loss = losses[0]
    final_loss = losses[-1]

    # Print real numbers explicitly
    print(f"\nMamba-SSSM Training - Initial Loss: {initial_loss:.6f}, Final Loss: {final_loss:.6f}")

    assert len(losses) == epochs
    assert final_loss < initial_loss, (
        f"Expected loss decrease: initial {initial_loss:.6f} vs final {final_loss:.6f}"
    )

    # Verify decide() interface on a sequence
    test_seq = [0.1, 0.9, 0.2]
    decisions = processor.decide(test_seq, tau=0.5)
    assert len(decisions) == len(test_seq)
    assert all(d in (0, 1) for d in decisions)


def test_mamba_sssm_cpu_device_explicit() -> None:
    # (c) Repeat the shape test with device="cpu" explicitly
    device = torch.device("cpu")
    model = MambaSSSMModel(in_features=1, d_model=12, d_state=6).to(device)

    x = torch.rand(3, 15, 1, device=device)
    logits = model(x)

    assert logits.shape == (3, 15)
    assert logits.device.type == "cpu"
