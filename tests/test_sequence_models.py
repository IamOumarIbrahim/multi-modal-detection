#   MMSAR - tests for the learned filters M5 and M6
#        Oumar Ibrahim U22200741

import pytest
import torch

from mmsar.sequence_models import GRUBaseline, MambaSSSM, SequenceFilter


@pytest.mark.parametrize("model", [MambaSSSM(), GRUBaseline()])
def test_output_shape(model):
    x = torch.rand(4, 20, 1)
    assert model(x).shape == (4, 20)


# one logit per frame and frame t must not depend on frames after t
@pytest.mark.parametrize("model", [MambaSSSM(), GRUBaseline()])
def test_causal(model):
    model.eval()
    x = torch.rand(1, 12, 1)
    y = x.clone()
    y[0, 8:, 0] = 1.0
    with torch.no_grad():
        a = model(x)[0, :8]
        b = model(y)[0, :8]
    assert torch.allclose(a, b)


@pytest.mark.parametrize("kind", ["mamba", "gru"])
def test_fit_and_decide(kind):
    seqs = [[0.1] * 10 + [0.9] * 10, [0.1] * 20]
    labels = [[0] * 10 + [1] * 10, [0] * 20]
    filt = SequenceFilter(kind=kind, seed=0)
    losses = filt.fit(seqs, labels, epochs=30, lr=0.05)
    assert len(losses) == 30
    assert losses[-1] < losses[0]

    dec = filt.decide(seqs[0], 0.5)
    assert len(dec) == 20
    for d in dec:
        assert d in (0, 1)


def test_same_seed_same_model():
    a = SequenceFilter(kind="gru", seed=7).predict_probs([0.2, 0.8, 0.5])
    b = SequenceFilter(kind="gru", seed=7).predict_probs([0.2, 0.8, 0.5])
    assert a == b
