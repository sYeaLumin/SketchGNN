# SketchGNN: Semantic Sketch Segmentation with Graph Neural Networks

We introduce *SketchGNN*, a convolutional graph neural network for semantic segmentation and labeling of freehand vector sketches. We treat an input stroke-based sketch as a graph, with nodes representing the sampled points along input strokes and edges encoding the stroke structure information. *SketchGNN* significantly improves the accuracy of the state-of-the-art methods for semantic sketch segmentation and has magnitudes fewer parameters than both image-based and sequence-based methods.

[[Paper]](https://arxiv.org/abs/2003.00678)

## Requirments

- Pytorch>=1.6.0
- pytorch_geometric>=1.6.1
- tensorboardX>=1.9

## How to use
**Test**

```bash
python evaluate.py --dataset SPG256 --class-name airplane --out-segment 4 --timestamp BEST --which-epoch bestloss
```

**Train**

````bash
python train.py --dataset SPG256 --class-name airplane --out-segment 4 --shuffle --stochastic
````

## License

MIT License

