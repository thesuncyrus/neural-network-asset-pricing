"""Existing modular-workflow settings; importing these does not change RNG state.

The project notebook retains its own explicit TrainConfig and execution settings;
it is a separate workflow, as documented in README.md.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    latent_dim: int = 8
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 20
    batch_size: int = 4096
    alpha_mse: float = 0.5
    beta_rank: float = 0.5
    lambda_rec: float = 0.05
    patience: int = 3


PANEL = dict(start="2001-01", end="2021-12", n_features=94)
EXPERIMENT = dict(first_year=2011, last_year=2021, seed=42)
BASELINES = dict(ridge_alpha=5.0, mlp_epochs=6, mlp_hidden_dim=64)
PORTFOLIO = dict(top_q=0.9, bot_q=0.1, min_n=30)
