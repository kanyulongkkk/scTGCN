
# scTGCN

scTGCN is a transfer learning method to integrate atlas-scale, heterogeneous collections of scRNA-seq and scATAC-seq data. scTGCN leverages information from annotated scRNA-seq data in a semi-supervised framework and uses graph convolutional network to simultaneously train labeled and unlabeled data, enabling label transfer and joint visualization in an integrative framework. 



## Installation

scTGCN can be obtained by simply clonning the github repository:

```

git clone https://github.com/kanyulongkkk/scTGCN.git
```

The following python packages are required to be installed before running scJoint:
`h5py`, `torch`, `itertools`, `scipy`, `numpy`,  `os`, `random`, `sys`, `time`, and `datetime`.


In terminal, run

```
python main.py
```

The output will be saved in `./output` folder.



=======
# scTGCN
Integration of unpaired single cell omics data by deep transfer graph convolutional network

