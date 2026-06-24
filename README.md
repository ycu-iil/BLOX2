## BLOX2

BLOX2 is an improved version of BLOX[^1]. The original implementations are available at https://github.com/tsudalab/BLOX.

[^1]: Terayama, K. and Sumita, M. and Tamura, R. and Payne, D. T. and Chahal, M. K. and Ishihara, S. and Tsuda, K. (2020). Pushing property limits in materials discovery via boundless objective-free exploration. <i>Chemical Science</i> [https://doi.org/10.1039/D0SC00982B](https://doi.org/10.1039/D0SC00982B)

## Installation (with tutorials)
1. Clone the repository
2. Install uv: https://docs.astral.sh/uv/getting-started/installation/
3. Restart the shell
4. Move to the repository root (e.g., cd BLOX2)
5. Run the following commands:
```bash
uv venv --python 3.11.11
source .venv/bin/activate
uv pip install blox2 numpy==2.4.1 pandas==2.3.3 matplotlib==3.10.8 scikit-learn==1.7.2 lightgbm==4.6.0 ipykernel==7.1.0
```

## Installation (package only)
```bash
uv pip install blox2 numpy==2.4.1 pandas==2.3.3 matplotlib==3.10.8 scikit-learn==1.7.2 lightgbm==4.6.0
```

## Tutorial
For a quick start, see `example_usage.ipynb`.

## Parameters

#### `SteinNoveltySelector`
`SteinNoveltySelector` selects candidates based on Stein novelty in the predicted objective space.  
It can optionally incorporate predictive uncertainty, input-space novelty, distributional predictions, and batch diversity penalties.

#### Required inputs
`observed_features`, `observed_values`, and `unobserved_features` can be provided as either `np.ndarray` or `pd.DataFrame`.

| Parameter             | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| `observed_features`   | Feature values of already observed samples                                  |
| `observed_values`     | Objective/property values of already observed samples.                          |
| `unobserved_features` | Feature values of candidate samples to be selected from.                        |
| `predictor`           | Predictor used to estimate objective/property values for unobserved candidates. |

#### Normalization and prediction settings
| Parameter             |         Default | Description                                                                                                                                   |
| --------------------- | --------------: | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `normalize_features`  |          `True` | Whether to normalize input features before prediction.                                                                                        |
| `value_normalization` | `"before_pred"` | How objective/property values are normalized. Options: `"before_pred"`, `"after_pred"`, `"mixed"`, or `"disable"`.                            |
| `pred_clip`           |          `None` | Valid ranges for predicted objective values, given as a list of `(min, max)` tuples. Cannot be used with `value_normalization="before_pred"`. |
| `sigma`               |           `1.0` | Gaussian kernel bandwidth used for Stein novelty calculation in objective space.                                                              |
| `n_obs_samples`       |          `None` | If the number of observed samples exceeds this value, a subset of observed samples is used for Stein novelty calculation.                     |

#### Uncertainty-aware selection
| Parameter                      |  Default | Description                                                                                                                                                |
| ------------------------------ | -------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_uncertainty`              |  `False` | Whether to combine predictive uncertainty with Stein novelty.                                                                                              |
| `uncertainty_ratio`            |    `0.5` | Weight of the uncertainty score. The final score is `(1 - uncertainty_ratio) * standardized Stein novelty + uncertainty_ratio * standardized uncertainty`. |
| `uncertainty_aggregation_type` | `"mean"` | How to aggregate uncertainty over objectives/features. Options: `"mean"`, `"max"`, or `"l2"`.                                                              |
| `print_uncertainty`            |  `False` | Whether to print uncertainty information during selection.                                                                                                 |

#### Input-space Stein novelty
| Parameter                    |  Default | Description                                                                                                                                                                     |
| ---------------------------- | -------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_input_stein_novelty`    |  `False` | Whether to also use Stein novelty in the input feature space.                                                                                                                   |
| `input_stein_novelty_ratio`  |    `0.5` | Weight of input-space Stein novelty. The final score is `(1 - input_stein_novelty_ratio) * output-space Stein novelty + input_stein_novelty_ratio * input-space Stein novelty`. |
| `input_stein_pca_dim`        |   `None` | If set to a positive integer, PCA is applied to the processed input features before input-space Stein novelty or Stein-based batch penalty calculation.                         |
| `input_stein_sigma`          | `"auto"` | Gaussian kernel bandwidth for input-space Stein novelty. If `"auto"`, the mean pairwise distance is used.                                                                       |
| `input_stein_auto_n_samples` |  `10**5` | Maximum number of sample pairs used to estimate the mean pairwise distance when `input_stein_sigma="auto"`.                                                                     |

#### Batch diversity penalty
| Parameter                    |   Default | Description                                                                                                                                               |
| ---------------------------- | --------: | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_batch_penalty`          |   `False` | Whether to penalize candidates that are too close to already selected candidates in the same batch.                                                       |
| `batch_penalty_ratio`        |     `0.5` | Weight of the batch penalty. The batch score is `(1 - batch_penalty_ratio) * standardized base score - batch_penalty_ratio * standardized batch penalty`. |
| `batch_penalty_type`         | `"stein"` | Type of batch penalty. Options: `"stein"`, `"distance"`, `"simhash"`, or `"simhash_min_hamming"`.                                                         |
| `batch_penalty_cutoff_ratio` |     `0.0` | Skips batch-penalty calculation for low-scoring candidates within each chunk.                                                                             |
| `batch_penalty_simhash_bits` |       `8` | Number of SimHash bits used when `batch_penalty_type` is `"simhash"` or `"simhash_min_hamming"`. Must be at most 64.                                      |

#### Distributional prediction
| Parameter                   |  Default | Description                                                                                 |
| --------------------------- | -------: | ------------------------------------------------------------------------------------------- |
| `use_distribution`          |  `False` | Whether to use predictive distributions from `predictor.pred_samples()`.                  |
| `distribution_pooling_type` | `"mean"` | How to aggregate Stein novelty scores over predicted samples. Options: `"mean"` or `"max"`. |

#### Misc
| Parameter                | Default | Description                                                                                        |
| ------------------------ | ------: | -------------------------------------------------------------------------------------------------- |
| `chunk_size`             |   `256` | Number of candidates processed in one chunk during chunked Stein novelty calculation.              |
| `verbose_plot_dir`       |  `None` | If set, saves diagnostic plots of predicted values and selected candidates at each selection step. |

## Predictor
Although `LightGBMPredictor` was mainly used in our experiments, BLOX2 is not limited to LightGBM. Any predictor can be used by implementing the `Predictor` class. The package also includes sklearn-based predictor implementations, such as `RandomForestPredictor` and `SVRPredictor`, although these implementations do not support uncertainty estimation. When using these predictors, the `lightgbm` dependency can be omitted.

## License

The BLOX2 source code is licensed under the MIT License.

The files under `data/` are provided for reproducibility and are not covered by the MIT License. See `data/README.md` for the corresponding attribution and license information.
