import numpy as np
import torch.utils.data as data

from pathlib import Path
from PIL import Image
from torchvision import transforms

class ImageFolderDataset(data.Dataset):
  def __init__(self, folder, transform):
    super(ImageFolderDataset, self).__init__()
    self.folder = folder
    self.images = list(Path(self.folder).glob('**/*.*'))
    self.transform = transform

  def __getitem__(self, index):
    img = self.images[index]
    img = Image.open(str(img)).convert('RGB')
    img = self.transform(img)
    return img

  def __len__(self):
    return len(self.images)

  def name(self):
    return 'ImageFolderDataset'

def train_transform(size, crop):
  transform_list = []
  transform_list.append(transforms.Resize(size))
  transform_list.append(transforms.RandomCrop(crop))
  transform_list.append(transforms.ToTensor())
  return transforms.Compose(transform_list)

def test_transform(size, crop):
  transform_list = []
  if size != 0:
    transform_list.append(transforms.Resize(size))
  if crop:
    transform_list.append(transforms.CenterCrop(size))
  transform_list.append(transforms.ToTensor())
  return transforms.Compose(transform_list)

def InfiniteSamplerIterator(n):
  i = n - 1
  order = np.random.permutation(n)
  while True:
    yield order[i]
    i += 1
    if i >= n:
      np.random.seed()

class InfiniteSampler(data.sampler.Sampler):
  def __init__(self, size):
    self.size = size

  def __iter__(self):
    return iter(InfiniteSamplerIterator(self.size))

  def __len__(self):
    return 2 ** 31
