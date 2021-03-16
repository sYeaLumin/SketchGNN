# SketchGNN: Semantic Sketch Segmentation with Graph Neural Networks

## Overview

## Requirments

- Pytorch>=1.6.0
- pytorch_geometric>=1.6.1
- tensorboardX>=1.9

## How to use
### Test

`python evaluate.py --dataset SPG256 --class-name airplane --out-segment 4 --timestamp BEST --which-epoch bestloss`



### Train

`python train.py --dataset SPG256 --class-name airplane --out-segment 4 --shuffle --stochastic`

## Citation

