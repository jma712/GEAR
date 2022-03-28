# GEAR-WSDM22:  Node Representation Learning for Graph Counterfactual Fairness

Code for the WSDM 2022 paper [*Learning Fair Node Representations with Graph Counterfactual Fairness*.](https://arxiv.org/pdf/2201.03662.pdf)

## Environment
```
Python 3.6
Pytorch 1.6.0
Sklearn 0.22
Numpy 1.18.3
Torch-geometric 1.3.0 
```

## Dataset
Datasets can be found in ```./datasets```

## Run Experiment
### Learning node representation
```
python main.py --experiment_type train
```
The subgraphs will be generated under ```./graphFair_subgraph/``` at the first time of running. If the files already exist, the subgraph data will be directly loaded. The true counterfactuals for evaluation are in ```./graphFair_subgraph/cf/```, and the augmented data is in ```./graphFair_subgraph/aug/```. The trained GEAR model can be saved in './models_save/'

### Refenrences
The code is the implementation of this paper:
```
[1] J. Ma, R. Guo, M. Wan, L. Yang, A. Zhang, and J. Li. Learning fair node representations with graph counterfactual fairness. In Proceedings of the 15th WSDM, 2022
```
Acknowledgement: The code in this work is developed based on part of the code in the following papers:
```
[2] Chirag Agarwal, Himabindu Lakkaraju, and Marinka Zitnik. Towards a unified framework for fair and stable graph representation learning. arXiv preprint arXiv:2102.13186, 2021.
[3] Jiao Y, Xiong Y, Zhang J, et al. Sub-graph contrast for scalable self-supervised graph representation learning[C]//2020 IEEE International Conference on Data Mining (ICDM). IEEE, 2020: 222-231.
[4] Kipf T N, Welling M. Variational graph auto-encoders[J]. arXiv preprint arXiv:1611.07308, 2016.
```
# GEAR
