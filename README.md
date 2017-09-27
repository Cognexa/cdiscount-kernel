# Open tensorflow kernel for Cdiscount’s Image Classification Challenge
Open **cxflow-tensorflow** kernel for [**Cdiscount’s Image Classification Challenge**](https://www.kaggle.com/c/cdiscount-image-classification-challenge) Kaggle competition.

Start training on multiple GPUs with **tensorflow** right away!

Works on Linux with Python 3.5+.

Features:
- CLI data download
- Data validation with SHA256 hash
- Simple data visualization
- Train-Valid splitting
- Low memory footprint data streams
- GPU-CPU parallelism
- Base VGG-like convnet
- Multi-GPU training with a single argument!
- TensorBoard training tracking
- Model prediction and submission

## Quick start
Install tensorflow and 7z.

Clone repo and install the requirements
```
git clone https://github.com/Cognexa/cdiscount-kernel && cd cdiscount-kernel
pip3 install -r requirements.txt --user
```

Download dataset with kaggle-cli (this may take a while, 3 hours in my case)
```
# requires >57Gb of free space
KG_USER="<YOUR KAGGLE USERNAME" KG_PASS="<YOUR KAGGLE PASSWORD>" cxflow dataset download cdc
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
cxflow train cdc model.n_gpus=<NUMBER OF GPUS TO USE>
```

Observe the training with TensorBoard (note: a summary is written only after each epoch)
```
tensorboard --logdir=log
```

Obtain predictions for submission:
```
cxflow predict log/<DIR> log/<DIR> model.restore_model_name=<CHECKPOINT NAME>
```

With only one checkpoint in the output directory, this simplifies to:
```
cxflow predict log/<DIR>
```

## UPDATE [0.65]
Main features:
- XCeption net (https://arxiv.org/abs/1610.02357)
- Fast random access

Resize the data to `dataset.size` with
```
cxflow dataset resize cdc/xception.yaml
```

(this may take a few hours)

Run the training with
```
cxflow train cdc/xception.yaml
```

Training procedure that reached 0.65:
- Train with original size, LR 0.0001, 4 middle flow repeats until stalled
- Fine-tune with 128x128, LR 0.0001, 0.5 dropout, 0.00001 weight decay until stalled
- Fine-tune as above but with LR 0.00001 (10x smaller)

Tips:
- Use small images right away
- The final GlobalAveragePooling may be a bottleneck
- Net does not overfit so far, no augmentations needed

## About
This kernel is written in [cxflow-tensorflow](https://github.com/Cognexa/cxflow-tensorflow), a plugin for [cxflow](https://github.com/Cognexa/cxflow) framework. Make sure you check it out!

A simple submission script will be added soon, stay tuned!
