"""Embodiment-conditioned MLP modules for N1.6 action head.

Ported from Isaac-GR00T gr00t/model/modules/embodiment_conditioned_mlp.py.
Contains: CategorySpecificLinear, CategorySpecificMLP, MultiEmbodimentActionEncoder.
"""

import torch
from torch import nn
import torch.nn.functional as F


def swish(x):
    """Swish activation function."""
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Produces a sinusoidal encoding of shape (B, T, w) given timesteps of shape (B, T)."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)
        return enc


class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights and biases for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

    def expand_action_dimension(
        self, old_action_dim, new_action_dim, expand_input=False, expand_output=False
    ):
        if new_action_dim <= old_action_dim:
            raise ValueError(
                f"New action dim {new_action_dim} must be larger than old action dim {old_action_dim}"
            )

        if expand_input and self.W.shape[1] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim
            new_W_parts = [self.W] * repeat_times
            if remainder > 0:
                new_W_parts.append(self.W[:, :remainder, :])
            new_W = torch.cat(new_W_parts, dim=1)
            self.W = nn.Parameter(new_W)

        if expand_output and self.W.shape[2] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim
            new_W_parts = [self.W] * repeat_times
            if remainder > 0:
                new_W_parts.append(self.W[:, :, :remainder])
            new_W = torch.cat(new_W_parts, dim=2)
            self.W = nn.Parameter(new_W)

            if self.b.shape[1] == old_action_dim:
                new_b_parts = [self.b] * repeat_times
                if remainder > 0:
                    new_b_parts.append(self.b[:, :remainder])
                new_b = torch.cat(new_b_parts, dim=1)
                self.b = nn.Parameter(new_b)


class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        self.layer2.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=False, expand_output=True
        )


class MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with multi-embodiment support and sinusoidal positional encoding."""

    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        self.W1.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=True, expand_output=False
        )
