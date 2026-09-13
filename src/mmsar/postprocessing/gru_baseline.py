"""Pure-PyTorch 1-Layer GRU sequence model for temporal post-processing baseline."""

from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmsar.postprocessing.base import PostProcessor


class GRUModel(nn.Module):
    """Minimal 1-layer GRU sequence classifier for temporal post-processing."""

    def __init__(self, in_features: int = 1, hidden_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.in_proj = nn.Linear(in_features, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, in_features) or (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = F.silu(self.in_proj(x))
        out, _ = self.gru(h)
        logits = self.head(out).squeeze(-1)
        return logits


class GRUPostProcessor:
    """PostProcessor adapter wrapping the trainable GRU model."""

    def __init__(
        self,
        in_features: int = 1,
        hidden_dim: int = 16,
        num_layers: int = 1,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = GRUModel(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

    def decide(self, confidence: Sequence[float], tau: float) -> list[int]:
        """Produce binary alarm decisions from confidence sequence at threshold tau."""
        if len(confidence) == 0:
            return []
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(confidence, dtype=torch.float32, device=self.device).view(1, -1, 1)
            logits = self.model(x).squeeze(0)
            probs = torch.sigmoid(logits)
            decisions = (probs >= tau).long().cpu().tolist()
        return decisions

    def fit(
        self,
        sequences: Sequence[Sequence[float]],
        labels: Sequence[Sequence[int]],
        epochs: int = 50,
        lr: float = 0.01,
    ) -> list[float]:
        """Train the GRU model on sequences and per-frame binary labels."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        max_len = max(len(s) for s in sequences)
        batch_size = len(sequences)

        x_pad = torch.zeros(batch_size, max_len, 1, dtype=torch.float32, device=self.device)
        y_pad = torch.zeros(batch_size, max_len, dtype=torch.float32, device=self.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, (seq, lab) in enumerate(zip(sequences, labels)):
            length = len(seq)
            x_pad[i, :length, 0] = torch.tensor(seq, dtype=torch.float32)
            y_pad[i, :length] = torch.tensor(lab, dtype=torch.float32)
            mask[i, :length] = True

        losses: list[float] = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.model(x_pad)
            loss = criterion(logits[mask], y_pad[mask])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        return losses
