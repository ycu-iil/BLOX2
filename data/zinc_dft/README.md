## feature_list_mfp2048_rdd.npz

This file contains precomputed feature values for each molecule. To load this feature file, use the `load_features()` function.

#### Example:
```python
from blox2.utils import load_features
features_df = load_features("data/zinc_dft/feature_list_mfp2048_rdd.npz")
```
