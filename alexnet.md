# ImageNet Classification with Deep Convolutional Neural Networks
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.

## Models

Using Dataset

| Implementation | Accuracy | Weights | Memory | Conv Ops | etc |
|---|---|---|---|---|---|
| Keras |   |  35,659,688 | 35,659,688 * 4bytes |   |    |
| Tensorflow |   | 71,324,339  | 71,324,339 * 4bytes |   |  [link](https://github.com/YBIGTA/DL_Models/blob/master/Alexnet%20(Tensorflow).ipynb)  |
| Pytorch (CIFAR 10) | 89.2%  | 58.3M  | 5.397GB |   |[link](https://github.com/Jooong/DLCV/blob/master/classification/models.py#L71)|

## Tip & Trick

| name | for What | reference |
|---|---|---|
| Relu | faster train / *training deeper network* | [Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.6419&rep=rep1&type=pdf) |
| LRN  | better generalization (replaced with BN later)  | [LRN in caffe](http://caffe.berkeleyvision.org/tutorial/layers/lrn.html) |
| Overlapping Pooling | to avoid overfitting | - |
| Drop Out | to avoid overfitting | [Paper](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) |
| Color PCA augmentation | to avoid overfitting, data augmentation | - |


## Error of paper
- 224x224 is actually 227x227
