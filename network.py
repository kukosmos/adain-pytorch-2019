import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.conv1_1 = nn.Conv2d(3, 3, (1, 1))
    self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv1_3 = nn.Conv2d(3, 64, (3, 3))
    self.relu1_4 = nn.ReLU()
    self.pad1_5 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv1_6 = nn.Conv2d(64, 64, (3, 3))
    self.relu1_7 = nn.ReLU()

    self.pool2_1 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
    self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv2_3 = nn.Conv2d(64, 128, (3, 3))
    self.relu2_4 = nn.ReLU()
    self.pad2_5 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv2_6 = nn.Conv2d(128, 128, (3, 3))
    self.relu2_7 = nn.ReLU()

    self.pool3_1 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
    self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv3_3 = nn.Conv2d(128, 256, (3, 3))
    self.relu3_4 = nn.ReLU()
    self.pad3_5 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv3_6 = nn.Conv2d(256, 256, (3, 3))
    self.conv3_7 = nn.ReLU()
    self.pad3_8 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv3_9 = nn.Conv2d(256, 256, (3, 3))
    self.relu3_10 = nn.ReLU()
    self.pad3_11 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv3_12 = nn.Conv2d(256, 256, (3, 3))
    self.relu3_13 = nn.ReLU()

    self.pool4_1 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
    self.pad4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv4_3 = nn.Conv2d(256, 512, (3, 3))
    self.relu4_4 = nn.ReLU()

  def forward(self, x):
    x = self.conv1_1(x)
    x = self.pad1_2(x)
    x = self.conv1_3(x)
    output_0 = self.relu1_4(x)

    x = self.pad1_5(output_0)
    x = self.conv1_6(x)
    x = self.relu1_7(x)
    x = self.pool2_1(x)
    x = self.pad2_2(x)
    x = self.conv2_3(x)
    output_1 = self.relu2_4(x)

    x = self.pad2_5(output_1)
    x = self.conv2_6(x)
    x = self.relu2_7(x)
    x = self.pool3_1(x)
    x = self.pad3_2(x)
    x = self.conv3_3(x)
    output_2 = self.relu3_4(x)

    x = self.pad3_5(output_2)
    x = self.conv3_6(x)
    x = self.conv3_7(x)
    x = self.pad3_8(x)
    x = self.conv3_9(x)
    x = self.relu3_10(x)
    x = self.pad3_11(x)
    x = self.conv3_12(x)
    x = self.relu3_13(x)
    x = self.pool4_1(x)
    x = self.pad4_2(x)
    x = self.conv4_3(x)
    output_3 = self.relu4_4(x)

    return output_0, output_1, output_2, output_3

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()

  def forward(self):
    pass

class AdaIN(nn.Module):
  def __init__(self):
    super(AdaIN, self).__init__()

  def forward(self):
    pass

  def save(self):
    pass

  def load(self):
    pass
