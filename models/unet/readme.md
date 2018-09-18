# U-Net: Convolutional Networks for Biomedical Image Segmentation
Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)

## Models

| Implementation | Dataset| Accuracy | Weights | Memory | Conv Ops | etc |
|---|---|---|---|---|---|---|
| Keras |   |   |  |  |   |    |
| Keras (Google Colab) |   |   | 31,030,658 | 124.12 MB |   |   |
| Pytorch |  |  |  |  |  | |

## Tip & Trick

| name | for What | reference |
|---|---|---|
| weight map  | compensate different frequency of pixels from a certain class | - |
| elastic distortion | data augmentation | [paper - section2](http://cognitivemedium.com/assets/rmnist/Simard.pdf) |
| bicubic interpolation | per-pixel displacement | [wikipedia](https://en.wikipedia.org/wiki/Bicubic_interpolation) |


## Error of paper
- add this if any errors
