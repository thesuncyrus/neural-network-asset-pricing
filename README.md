# Neural Network Modeling

**Review guide:** [Data and execution](DATA_SETUP.md) · [Archived vs computed results and fixes](RESULTS_PROVENANCE.md) · [Workflow map](docs/WORKFLOW.md).


Purdue University
January–March 2026 · Chau Nguyen

## Project

- Trained a PyTorch supervised autoencoder that maps firm characteristics to latent return-predicting factors.
- Assessed it against Ridge, PCA + Ridge, and MLP baselines through expanding-window out-of-sample backtests.
- Interpreted latent factors via characteristic loadings and evaluated long-short decile portfolio returns.

## Historical and corrected workflows

`notebooks/historical_supervised_autoencoder.ipynb` preserves the recovered architecture, preprocessing order, original losses, splits and interpretation retraining. The modular `analysis.py` workflow is explicitly a corrected implementation: it moves imputation before target filtering, stabilizes ranking loss, weights validation batches, guards undefined metrics and reuses the fitted encoder. These can change results and are not historical replay. The full historical replay completed on 2026-11-16 using the private 252-file panel. Saved original tables remain in `results/`; newly computed annual metrics are separately labeled in `results/replay-2026-11-16/`. SAE mean Rank IC is 0.080023 in this run versus the saved 0.080381.

## Data and methods

The analysis uses monthly U.S. equity observations from 2001–2021. The notebook records 1,520,078 initial rows and 1,027,681 rows after excluding missing excess-return targets. Model inputs are the 94 `characteristic_*` columns; the macroeconomic and industry columns are not used as model inputs.

The separately corrected pipeline fills missing characteristics with same-month cross-sectional medians before filtering unavailable targets, then zero if an entire monthly feature is missing. Each scaler is fit only on the training partition. For test year t, models train through t−2; neural-network early stopping uses t−1. Tests span 2011–2021 (132 months).

The supervised autoencoder maps 94 inputs through a 64-unit hidden layer into eight latent factors. It has a return-prediction head and a reconstruction decoder. Its training objective weights return MSE by 0.5, within-month pairwise ranking loss by 0.5, and reconstruction MSE by 0.05. Early stopping maximizes validation Rank IC. Ridge, PCA + Ridge (eight components), and MLP provide the comparisons.

`oos_r2` uses the test-year return mean as its denominator benchmark. The table averages each model's annual metrics across eleven test years. Rank IC is the mean monthly Spearman correlation. Higher ranking performance does not imply smaller return-prediction error or better portfolio performance.

## Decile portfolio evaluation

Each month, take the equal-weight mean realized return of securities at or above the 90th prediction percentile and subtract the corresponding mean at or below the 10th percentile. Ties at either boundary are included. The current runnable notebook and `portfolio.py` use top/bottom 10%. The cleaned historical notebook also contains earlier quintile experiments; those are not the source of the decile table.

The portfolio table covers all 132 test months. Sharpe ratios in this table are monthly (mean monthly long-short return divided by its sample standard deviation), not annualized. Cumulative return is the product of one plus each monthly long-short return, minus one. Returns are gross of trading costs and borrowing costs.

In the saved original results, the SAE has the strongest average Rank IC in the model comparison. Ridge has the strongest long-short decile return and monthly Sharpe ratio in the portfolio comparison. All four mean R² values are negative against the stated test-year-mean benchmark.

## Factor interpretation

Characteristic loadings are approximated by multiplying the encoder's two linear weight matrices. Because the encoder contains ReLU activations, this is a weight-based interpretation, not an exact marginal effect. `results/factor-loadings.csv` contains the twenty entries displayed in the notebook: the ten highest absolute weights for each of factors 0 and 1. Running the analysis exports all eight factors. The notebook also includes the 2021 monthly cross-sectional factor-premium estimates and their plots.

## Files and run instructions

- `supervised_autoencoder_analysis.ipynb`: executable review notebook calling the tested modules. `results/` separately retains saved aggregate outputs.
- `data.py`: monthly Parquet loader with sample coverage and duplicate-key checks.
- `analysis.py`: expanding-window runner, artifacts and metadata; `models.py`: importable PyTorch models, baselines, losses and metrics; `predict.py`: checkpoint inference.
- `portfolio.py`: monthly top/bottom-decile construction.
- `plot.py`: creates the website figures from the included aggregate tables.
- `results/`: aggregate tables from the supplied notebook's saved outputs.
- `figures/`: SVG figures generated from saved aggregate results.

Install dependencies with `python -m pip install -r requirements.txt`.

Replay the historical notebook with:

    python run_historical.py --data-dir /path/to/monthly-parquet --output-dir artifacts/historical

Run the separately corrected analysis with:

    python analysis.py --data-dir /path/to/monthly-parquet --output-dir artifacts

The input directory must contain the 2001–2021 monthly Parquet files named `YYYYMM_0.parquet`; subdirectories are supported. Open the notebook from this folder and set its data path to use the notebook workflow. PyTorch can use CUDA when available; CPU execution is also supported.

Create the website charts without training models:

    python plot.py --output-dir figures

Source material: `MGMT475_Project_final.ipynb`, the course dataset documentation, and the course project guidelines. The dataset documentation attributes the underlying sources to CRSP, Compustat/WRDS, and the asset-pricing literature. Raw equity records are not included in the package.


## Portfolio

[Project overview and visualizations](https://chau-nguyen-research.gabledmango1.chatgpt.site/neural-network.html) · [Chau Nguyen on LinkedIn](https://www.linkedin.com/in/chaunguyen2811)

## Validation

See [executed checks and limits](docs/VALIDATION.md) and [result provenance](RESULTS_PROVENANCE.md).
