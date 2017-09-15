# cdiscount-kernel
Open **cxflow-tensorflow** kernel for [**Cdiscount’s Image Classification Challenge**](https://www.kaggle.com/c/cdiscount-image-classification-challenge) Kaggle competition.

Start training on multiple GPUs with **tensorflow** right away!

## Quick start
Clone repo and install the requirements
```
git clone https://github.com/Cognexa/cdiscount-kernel && cd cdiscount-kernel
pip3 install -r requirements.txt --user
```

Download dataset with kaggle-cli (this may take a while, 3 hours in my case)
```
# requires >57Gb of free space
mkdir data && cd data
kg download -u '<YOUR KAGGLE USERNAME>' -p '<YOUR KAGGLE PASSWORD>' -c 'cdiscount-image-classification-challenge'
```

Validate your download and see the example data:
```
# in the root directory (cdiscount-kernel)
cxflow dataset validate cdc
cxflow dataset show cdc
# now see the newly created visual directory
```

Create a random validation split with 10% of the data and start training:
```
cxflow dataset split cdc
cxflow train cdc model.n_gpus=<NUMBER OF GPUS TO USE
```

## About
This kernel is written in [cxflow-tensorflow](https://github.com/Cognexa/cxflow-tensorflow), a plugin for [cxflow](https://github.com/Cognexa/cxflow) framework. Make sure you check it out!

A simple submission script will be added soon, stay tuned!
