## BLOX2

BLOX2 is an improved version of BLOX[^1]. The original implementations are available at https://github.com/tsudalab/BLOX.

[^1]: Terayama, K. and Sumita, M. and Tamura, R. and Payne, D. T. and Chahal, M. K. and Ishihara, S. and Tsuda, K. (2020). Pushing property limits in materials discovery via boundless objective-free exploration. <i>Chemical Science</i> [https://doi.org/10.1039/D0SC00982B](https://doi.org/10.1039/D0SC00982B)

## Installation (package only)
```bash
uv pip install blox2 numpy==2.4.1 pandas==2.3.3 matplotlib==3.10.8 scikit-learn==1.7.2 lightgbm==4.6.0
```

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

## Tutorial
For a quick start, see `example_usage.ipynb`.