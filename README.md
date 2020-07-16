# QuickNat - Pytorch implementation

Tool: QuickNAT: Segmenting Kidney from CT scans
-----------------------------------------------------------

 

## Getting Started
### Dataset
The dataset used for the project is KiTS19 kidney segmentation challenge [data](https://kits19.grand-challenge.org/data/). The functions 
for loading the dataset are defined in  [kits_data_utils.py](utils/kits_data_utils.py) and [data_utils.py](utils/data_utils.py).


### Pre-requisites
The prerequisites for running and deploying the project are defined in requirements.txt

### Training your model

```
python run.py --mode=train
```

### Evaluating your model

```
python run.py --mode=eval
```
