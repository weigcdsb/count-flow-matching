# Flow Matching for Count Data

The Python implementation of:

> Ganchao Wei and John Pearson. **Flow Matching for Count Data.** *NeurIPS*, 2026. [Preprint](https://arxiv.org/abs/2605.07746)

Count-FM learns birth and death rates on nonnegative integer counts using a
signed-binomial bridge, with independent or minibatch OT endpoint coupling.

## Setup

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Install PyTorch for your CPU/CUDA environment first if needed. For scRNA
notebooks, also install `.[scrna]`; for neural-data preparation, `.[neural]`.
Comparison methods with external repositories need their original environments
and paths configured in the notebook. Dirichlet-FM expects the upstream
`DirichletFM` checkout on `PYTHONPATH`.

## Layout

| Location | Contents |
| --- | --- |
| `count_fm/bridge.py`, `coupling.py`, `training.py`, `sampling.py` | Core method |
| `count_fm/models/` | MLP, transformer, U-Net and conditional models |
| `count_fm/utils/` | Checkpoints, metrics, datasets and probability paths |
| `count_fm/baselines/` | Comparison methods and scRNA adapters |
| `count_fm/experiments/` | Experiment-specific training and sampling |
| `notebooks/` | Toy simulation, scRNA and neural experiments |

Start with `notebooks/1_simulation_toy_comparison_patched.ipynb`. The numbered
notebooks retain the original experiment order. 

## Citation

```bibtex
@inproceedings{wei2026flow,
  title     = {Flow Matching for Count Data},
  author    = {Wei, Ganchao and Pearson, John},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.07746}
}
```
