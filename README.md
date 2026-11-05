# CDN: Cauchy Displacement Networks

Code for the paper *Anomalies as Residues: Cauchy Displacement Networks*.

CDN is a tabular anomaly detector. During training, each normal point is regressed onto the mean of its *k* nearest training neighbours by a single-hidden-layer network with Cauchy activations. A query is scored by the length of the displacement between the query and the network output, so scoring takes one forward pass, needs no neighbour search and does not use the training data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from cdn import CDN

model = CDN().fit(X_train)                # X_train: normal data, shape (n, d)
scores = model.decision_function(X_test)  # larger score = more anomalous
```

The default parameters are the single configuration used for all datasets in the paper (`hidden=128`, `k=5`, `steps=2000`). `CDN` follows the [PyOD](https://github.com/yzhao062/pyod) detector interface, so `predict` and the other PyOD methods are also available.

## Demo

`demo.ipynb` trains CDN on the `breastw` dataset of [ADBench](https://github.com/Minqi824/ADBench) with the protocol of the paper and reproduces the reported AUC-ROC and AUC-PR.

## Results

`results/cdn_adbench46.csv` lists the AUC-ROC and AUC-PR of CDN on the 46 ADBench datasets used in the paper, as mean and standard deviation over five seeds (42, 123, 456, 789, 1024). The protocol is the one in `demo.ipynb`: the first half of the normal samples of each dataset is the training set, and the test set contains the remaining normal samples and all anomalies.

## MOD baseline

`baselines/mod.py` contains the semi-supervised adaptation of the mean-shift outlier detector (MOD) of Yang et al. (2021) used as a baseline in the paper, in which each query is shifted towards the mean of its *k* nearest neighbours in the fixed training set.

## Citation

```bibtex
@misc{cao2026cdn,
  title  = {Anomalies as Residues: Cauchy Displacement Networks},
  author = {Cao, Yang and Guo, Zihao},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT
