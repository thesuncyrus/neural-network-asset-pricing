# Workflow map

Project: Neural Network Modeling | MGMT 47500: Machine Learning for Business, Purdue University, January–March 2026. The CV's three project claims map to:

| Documented work | Current implementation | Evidence |
| --- | --- | --- |
| Supervised autoencoder for return-predicting latent factors | `models.py:SupervisedAutoencoder`, `train_one_model` | Retained notebook architecture, prediction/ranking/reconstruction objective |
| Ridge, PCA + Ridge and MLP expanding-window baselines | `data.py:temporal_split`, `analysis.py:run_backtest` | Same fit/validation/test years for all models; annual metrics and splits |
| Characteristic loadings and decile portfolios | `encoder_effective_weights`, `portfolio.py` | Top/bottom 10%, monthly gross returns; approximate loadings |

The modular executable path is a reconstruction/refactor prepared with AI assistance from the uploaded notebook. Archived aggregates and newly computed outputs have separate paths and provenance. No saved output is relabelled as a successful full-data rerun.
