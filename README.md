# AdaIN
PyTorch implementation of [this paper](https://arxiv.org/abs/1703.06868).
Original implementation can be found at [here](https://github.com/xunhuang1995/AdaIN-style).

## Clone
via ssh
```
$ git clone git@github.com:kukosmos/AdaIN-pytorch-2019.git
```
via https
```
$ git clone https://github.com/kukosmos/AdaIN-pytorch-2019.git
```

## Requirments
Install following requirments
* Pillow
* [pytorch](https://pytorch.org)
* [torchvision](https://pytorch.org)
* TensorboardX
* tqdm

## Train

### Prepare Dataset
First, download the dataset with given script.
For example, to download ```coco2017train``` dataset, type in the following command:
```
$ ./download.sh coco2017train
```
Detailed usage of ```download.sh``` script can be found as fallows:
```
$ ./download.sh --help
```

## Test

## References
* [1]: X. Huang and S. Belongie. "[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)", in ICCV, 2017.
* [2]: [PyTorch implementation of AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
* [3]: [COCO Dataset](http://cocodataset.org/#download)
* [4]: [Wikiart dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)
