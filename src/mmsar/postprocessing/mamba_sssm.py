"""Simplified pure-PyTorch Mamba-style Selective State-Space Sequence Model (SSSM)."""

from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmsar.postprocessing.base import PostProcessor


class SelectiveSSMLayer(nn.Module):
    """Simplified diagonal selective state-space layer in pure PyTorch."""

    def __init__(self, d_model: int = 16, d_state: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A is parameterized as log of diagonal elements to guarantee negative eigenvalues
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)))

        # Selective projections from input x_t to Delta, B, C
        self.x_proj = nn.Linear(d_model, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential selective scan forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        b, l, d = x.shape
        A = -torch.exp(self.A_log)  # (d_model, d_state)

        # Project x to Delta, B, C parameters per token
        proj = self.x_proj(x)  # (b, l, 1 + 2 * d_state)
        delta = proj[..., :1]  # (b, l, 1)
        B = proj[..., 1 : 1 + self.d_state]  # (b, l, d_state)
        C = proj[..., 1 + self.d_state :]  # (b, l, d_state)

        # Delta softplus
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_model)

        # Discretization and recurrent sequential scan
        # h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * x_t
        h = torch.zeros(b, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(l):
            dt_t = delta[:, t, :].unsqueeze(-1)  # (b, d_model, 1)
            dA_t = torch.exp(dt_t * A.unsqueeze(0))  # (b, d_model, d_state)
            B_t = B[:, t, :].unsqueeze(1)  # (b, 1, d_state)
            x_t = x[:, t, :].unsqueeze(-1)  # (b, d_model, 1)
            dB_t = dt_t * (x_t * B_t)  # (b, d_model, d_state)

            h = dA_t * h + dB_t  # (b, d_model, d_state)
            C_t = C[:, t, :].unsqueeze(1)  # (b, 1, d_state)
            y_t = torch.sum(h * C_t, dim=-1)  # (b, d_model)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (b, l, d_model)
        return self.out_proj(y)


class MambaSSSMModel(nn.Module):
    """Neural network composed of projection, selective SSM layer, and classification head."""

    def __init__(self, in_features: int = 1, d_model: int = 16, d_state: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.ssm = SelectiveSSMLayer(d_model=d_model, d_state=d_state)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, in_features) or (batch, seq_len)

        Returns:
            Logits tensor of shape (batch, seq_len)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = F.silu(self.in_proj(x))
        h = self.ssm(h)
        logits = self.head(h).squeeze(-1)
        return logits


class MambaSSSMPostProcessor:
    """PostProcessor adapter wrapping the trainable MambaSSSM model."""

    def __init__(
        self,
        in_features: int = 1,
        d_model: int = 16,
        d_state: int = 8,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = MambaSSSMModel(
            in_features=in_features,
            d_model=d_model,
            d_state=d_state,
        ).to(self.device)

    def decide(self, confidence: Sequence[float], tau: float) -> list[int]:
        """Produce binary alarm decisions from confidence sequence at threshold tau."""
        if len(confidence) == 0:
            return []
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(confidence, dtype=torch.float32, device=self.device).view(1, -1, 1)
            logits = self.model(x).squeeze(0)  # (seq_len,)
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
        """Train the model on sequences and per-frame binary labels.

        Args:
            sequences: Batch of confidence sequences.
            labels: Matching batch of binary target labels.
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            List of loss values per epoch.
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        max_len = max(len(s) for s in sequences)
        batch_size = len(sequences)

        # Pad sequences to uniform length
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
