import torch
import torch.nn as nn
import torchvision.models as models

from utils import adaptive_instance_normalization as adain
from utils import calc_mean_std

class Encoder(nn.Module):
  def __init__(self, pretrained=True):
    super(Encoder, self).__init__()

    vgg = models.vgg19(pretrained=pretrained)
    features = list(vgg.features.children())

    self.enc_0 = nn.Sequential(*features[:2])
    self.enc_1 = nn.Sequential(*features[2:7])
    self.enc_2 = nn.Sequential(*features[7:12])
    self.enc_3 = nn.Sequential(*features[12:21])

  def forward(self, x):
    output_0 = self.enc_0(x)
    output_1 = self.enc_1(output_0)
    output_2 = self.enc_2(output_1)
    output_3 = self.enc_3(output_2)

    return output_0, output_1, output_2, output_3

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()

    self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv2 = nn.Conv2d(512, 256, (3, 3))
    self.relu3 = nn.ReLU()
    self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
    self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv6 = nn.Conv2d(256, 256, (3, 3))
    self.relu7 = nn.ReLU()
    self.pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv9 = nn.Conv2d(256, 256, (3, 3))
    self.relu10 = nn.ReLU()
    self.pad11 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv12 = nn.Conv2d(256, 256, (3, 3))
    self.relu13 = nn.ReLU()
    self.pad14 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv15 = nn.Conv2d(256, 128, (3, 3))
    self.relu16 = nn.ReLU()
    self.upsample17 = nn.Upsample(scale_factor=2, mode='nearest')
    self.pad18 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv19 = nn.Conv2d(128, 128, (3, 3))
    self.relu20 = nn.ReLU()
    self.pad21 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv22 = nn.Conv2d(128, 64, (3, 3))
    self.relu23 = nn.ReLU()
    self.upsample24 = nn.Upsample(scale_factor=2, mode='nearest')
    self.pad25 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv26 = nn.Conv2d(64, 64, (3, 3))
    self.relu27 = nn.ReLU()
    self.pad28 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv29 = nn.Conv2d(64, 3, (3, 3))
    
  def forward(self, x):
    x = self.pad1(x)
    x = self.conv2(x)
    x = self.relu3(x)
    x = self.upsample4(x)
    x = self.pad5(x)
    x = self.conv6(x)
    x = self.relu7(x)
    x = self.pad8(x)
    x = self.conv9(x)
    x = self.relu10(x)
    x = self.pad11(x)
    x = self.conv12(x)
    x = self.relu13(x)
    x = self.pad14(x)
    x = self.conv15(x)
    x = self.relu16(x)
    x = self.upsample17(x)
    x = self.pad18(x)
    x = self.conv19(x)
    x = self.relu20(x)
    x = self.pad21(x)
    x = self.conv22(x)
    x = self.relu23(x)
    x = self.upsample24(x)
    x = self.pad25(x)
    x = self.conv26(x)
    x = self.relu27(x)
    x = self.pad28(x)
    x = self.conv29(x)

    return x

class AdaIN(nn.Module):
  def __init__(self, torchvision_encoder=True, training_mode=True):
    super(AdaIN, self).__init__()

    self.encoder = Encoder(pretrained=torchvision_encoder)
    self.decoder = Decoder()
    self.mse_loss = nn.MSELoss()

    self.training_mode = training_mode

    for p in self.encoder.parameters():
      p.requires_grad = False

  def forward(self, content, style, alpha=1.0, interpolation_weights=None):
    assert 0 <= alpha <= 1, '"alpha" should be between 0 and 1'

    f_content = self.encoder(content)[-1]
    f_style = self.encoder(style)
    if interpolation_weights is not None:
      t = adain(f_content, f_style[-1])
      t = torch.mm(interpolation_weights, t)
      f_content = f_content[0:1]
    else:
      t = adain(f_content, f_style[-1])
    t = alpha * t + (1 - alpha) * f_content

    g = self.decoder(t)

    if not self.training_mode:
      return g

    f_g = self.encoder(g)

    l_content = self.mse_loss(f_g[-1], t)
    
    l_style = 0
    for i in range(4):
      mean_s, std_s = calc_mean_std(f_style[i])
      mean_g, std_g = calc_mean_std(f_g[i])
      l_style += self.mse_loss(mean_s, mean_g) + self.mse_loss(std_s, std_g)

    return g, l_content, l_style

def save_AdaIn(model, path='AdaIN.pth', include_encoder=False):
  state_dict = {}

  if include_encoder:
    encoder_dict = model.encoder.state_dict()
    for key in encoder_dict.keys():
      encoder_dict[key] = encoder_dict[key].to(torch.device('cpu'))
    state_dict['encoder'] = encoder_dict
  
  decoder_dict = model.decoder.state_dict()
  for key in decoder_dict.keys():
    decoder_dict[key] = decoder_dict[key].to(torch.device('cpu'))
  state_dict['decoder'] = decoder_dict
  
  torch.save(state_dict, path)

def load_AdaIN(path='AdaIN.pth', training_mode=False):
  state_dict = torch.load(path)
  model = AdaIN(torchvision_encoder=('encoder' not in state_dict.keys()), training_mode=training_mode)
  try:
    model.encoder.load_state_dict(state_dict['encoder'])
  except KeyError:
    pass
  model.decoder.load_state_dict(state_dict['decoder'])
  return model
