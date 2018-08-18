# SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg

## Models

Using Dataset

| Implementation | Accuracy | Weights | Memory | Conv Ops | etc |
|---|---|---|---|---|---|
| Keras |   |   |  |   |    |
| TensorFlow |   |   |  |   |   |
| PyTorch |   |   | |   |   |

## Tip & Trick

| name | for What | reference |
| ---  | ---      |    ---    |
| Multi-scale featuremap | allow prediction at multiple scales |  -  |
| Default boxes | allow separately detect objects with different ratio |  'Anchor' in [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)  |
|  jaccard overlay  |  to get best box   |  a.k.a. IOU  |
|   Smooth L1 loss   |  loss function for bbox regression  |  [image](https://www.researchgate.net/publication/322582664/figure/fig5/AS:584361460121600@1516334037062/The-curve-of-the-Smooth-L1-loss.png)         |

## Error of paper
- 
