"""Tests for 1-layer GRU baseline post-processing module."""

import pytest
import torch
from mmsar.postprocessing.base import PostProcessor
from mmsar.postprocessing.gru_baseline import GRUModel, GRUPostProcessor


def test_gru_baseline_output_shape() -> None:
    batch_size = 4
    seq_len = 25
    hidden_dim = 16

    model = GRUModel(in_features=1, hidden_dim=hidden_dim)
    x = torch.rand(batch_size, seq_len, 1)
    logits = model(x)

    assert logits.shape == (batch_size, seq_len), (
        f"Expected logits shape {(batch_size, seq_len)}, got {logits.shape}"
    )


def test_gru_baseline_training_loss_decrease() -> None:
    torch.manual_seed(42)

    sequences = [
        [0.1, 0.2, 0.7, 0.8, 0.1, 0.9, 0.2, 0.8],
        [0.8, 0.9, 0.1, 0.2, 0.7, 0.8, 0.2, 0.1],
        [0.1, 0.1, 0.2, 0.2, 0.8, 0.9, 0.7, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.1, 0.2],
    ]
    labels = [[1 if val >= 0.6 else 0 for val in seq] for seq in sequences]

    processor = GRUPostProcessor(hidden_dim=16, device="cpu")
    assert isinstance(processor, PostProcessor)

    epochs = 40
    losses = processor.fit(sequences, labels, epochs=epochs, lr=0.03)

    initial_loss = losses[0]
    final_loss = losses[-1]

    assert len(losses) == epochs
    assert final_loss < initial_loss, (
        f"Expected loss decrease: initial {initial_loss:.6f} vs final {final_loss:.6f}"
    )

    test_seq = [0.1, 0.9, 0.2]
    decisions = processor.decide(test_seq, tau=0.5)
    assert len(decisions) == len(test_seq)
    assert all(d in (0, 1) for d in decisions)
