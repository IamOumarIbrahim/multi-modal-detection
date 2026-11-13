#   MMSAR - learned filters M5 (Mamba-SSSM) and M6 (GRU)
#        Oumar Ibrahim U22200741

import torch
import torch.nn as nn
import torch.nn.functional as F


# diagonal selective state space layer, scanned one frame at a time so it stays causal
class SelectiveSSM(nn.Module):
    def __init__(self, d_model=16, d_state=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A is stored as log so -exp(A_log) is always negative (stable)
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a.unsqueeze(0).repeat(d_model, 1)))

        # x -> delta, B, C
        self.x_proj = nn.Linear(d_model, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, d = x.shape
        A = -torch.exp(self.A_log)

        proj = self.x_proj(x)
        delta = proj[..., :1]
        B = proj[..., 1 : 1 + self.d_state]
        C = proj[..., 1 + self.d_state :]
        delta = F.softplus(self.dt_proj(delta))

        # h_t = exp(delta_t * A) * h_(t-1) + delta_t * B_t * x_t
        h = torch.zeros(b, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):
            dt = delta[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt * A.unsqueeze(0))
            Bt = B[:, t, :].unsqueeze(1)
            xt = x[:, t, :].unsqueeze(-1)
            h = dA * h + dt * (xt * Bt)

            Ct = C[:, t, :].unsqueeze(1)
            ys.append(torch.sum(h * Ct, dim=-1))

        y = torch.stack(ys, dim=1)
        return self.out_proj(y)


class MambaSSSM(nn.Module):
    def __init__(self, d_model=16, d_state=8):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.ssm = SelectiveSSM(d_model=d_model, d_state=d_state)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x is (batch, frames, 1), output is one logit per frame
        h = F.silu(self.in_proj(x))
        h = self.ssm(h)
        return self.head(h).squeeze(-1)


class GRUBaseline(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.in_proj = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.silu(self.in_proj(x))
        out, _ = self.gru(h)
        return self.head(out).squeeze(-1)


# wraps a model so it can be trained and used like the classical filters
class SequenceFilter:
    def __init__(self, kind="mamba", seed=42):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # seed before building the model so the init weights are the same every run
        torch.manual_seed(seed)
        if kind == "mamba":
            self.model = MambaSSSM(d_model=16, d_state=8).to(self.device)
        elif kind == "gru":
            self.model = GRUBaseline(hidden_dim=16).to(self.device)
        else:
            raise ValueError("kind must be mamba or gru")

    # full batch training, clips are padded and the padding is masked out of the loss
    def fit(self, sequences, labels, epochs=40, lr=0.01):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        max_len = 0
        for s in sequences:
            if len(s) > max_len:
                max_len = len(s)
        n = len(sequences)

        x = torch.zeros(n, max_len, 1, device=self.device)
        y = torch.zeros(n, max_len, device=self.device)
        mask = torch.zeros(n, max_len, dtype=torch.bool, device=self.device)
        for i in range(n):
            lenght = len(sequences[i])
            x[i, :lenght, 0] = torch.tensor(sequences[i], dtype=torch.float32)
            y[i, :lenght] = torch.tensor(labels[i], dtype=torch.float32)
            mask[i, :lenght] = True

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.model(x)
            loss = criterion(logits[mask], y[mask])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    # alert probability for every frame of one clip
    def predict_probs(self, seq):
        if len(seq) == 0:
            return []
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).view(1, -1, 1)
            logits = self.model(x).squeeze(0)
            probs = torch.sigmoid(logits).cpu().tolist()
        return probs

    def decide(self, seq, tau):
        out = []
        for p in self.predict_probs(seq):
            if p >= tau:
                out.append(1)
            else:
                out.append(0)
        return out
