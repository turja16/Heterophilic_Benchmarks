# Dataset Generation using GenCAT 
This is based on the repository from [GenCAT](https://github.com/seijimaekawa/empirical-study-of-GNNs)

To generate synthetic graphs based on Cora, under the `empirical-study-of-GNNs` folder, run the following command:
```python
python scripts/run_gencat_hetero_homo.py --dataset cora
```
For each homophily levels, 10 datasets are generated.
Datasets are placed in this folder.
