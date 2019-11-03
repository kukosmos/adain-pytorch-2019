import torch
import torch.nn as nn
import torchvision.models as models

from utils import adaptive_instance_normalization as adain
from utils import calc_mean_std

# encoder using vgg19 structure
class VGGEncoder(nn.Module):
  def __init__(self, pretrained=True):
    super(VGGEncoder, self).__init__()

    vgg = models.vgg19(pretrained=pretrained)
    features = list(vgg.features.children())

    # use 4 steps to encode the image
    self.enc_0 = nn.Sequential(*features[:2])
    self.enc_1 = nn.Sequential(*features[2:7])
    self.enc_2 = nn.Sequential(*features[7:12])
    self.enc_3 = nn.Sequential(*features[12:21])
    self.encoding_step = 4

  # use four internal feature vectors of encoder
  def forward(self, x):
    output_0 = self.enc_0(x)
    output_1 = self.enc_1(output_0)
    output_2 = self.enc_2(output_1)
    output_3 = self.enc_3(output_2)

    return output_0, output_1, output_2, output_3

# decoder that mirrors encoder
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

# main architecture
class AdaIN(nn.Module):
  def __init__(self, torchvision_encoder=True, training_mode=True):
    super(AdaIN, self).__init__()

    self.encoder = VGGEncoder(pretrained=torchvision_encoder)
    self.decoder = Decoder()
    self.mse_loss = nn.MSELoss()

    # if training mode, calcuate loss, otherwise, not
    self.training_mode = training_mode

    # do not modify parameters from encoder
    for p in self.encoder.parameters():
      p.requires_grad = False

  def forward(self, content, style, alpha=1.0, interpolation_weights=None):
    assert 0 <= alpha <= 1, '"alpha" should be between 0 and 1.'
    assert interpolation_weights is None or not self.training_mode, 'Interpolation is not supported while training.'

    # get the features from the content image and the style image
    f_content = self.encoder(content)[-1]
    f_style = self.encoder(style)
    if interpolation_weights is not None:
      assert not self.training_mode, 'Interpolation is only avaialble for testing.'
      # mix the features of style images with interpolation weights
      t = adain(f_content.expand_as(f_style[-1]), f_style[-1])
      orig_shape = t.shape
      t = torch.reshape(t, (t.shape[0], -1))
      interpolation_weights = interpolation_weights.unsqueeze(1).expand_as(t)
      t = torch.reshape(t * interpolation_weights, orig_shape)
      t = torch.sum(t, dim=0, keepdim=True)
    else:
      t = adain(f_content, f_style[-1])
    # adjust amount of stylization with alpha
    t = alpha * t + (1 - alpha) * f_content

    # create image
    g = self.decoder(t)

    # return image if not training
    if not self.training_mode:
      return g

    # get the features from generated image
    f_g = self.encoder(g)

    # calculate content loss
    l_content = self.mse_loss(f_g[-1], t)
    
    # calculate style loss
    l_style = 0
    for i in range(self.encoder.encoding_step):
      mean_s, std_s = calc_mean_std(f_style[i])
      mean_g, std_g = calc_mean_std(f_g[i])
      l_style += self.mse_loss(mean_s, mean_g) + self.mse_loss(std_s, std_g)

    return g, l_content, l_style

# save the model
def save_AdaIn(model, path='AdaIN.pth', include_encoder=False):
  state_dict = {}

  # include encoder if specified
  if include_encoder:
    # get the states of encoder
    encoder_dict = model.encoder.state_dict()
    for key in encoder_dict.keys():
      encoder_dict[key] = encoder_dict[key].to(torch.device('cpu'))
    state_dict['encoder'] = encoder_dict
  
  # get the states of decoder
  decoder_dict = model.decoder.state_dict()
  for key in decoder_dict.keys():
    decoder_dict[key] = decoder_dict[key].to(torch.device('cpu'))
  state_dict['decoder'] = decoder_dict
  
  # save the model
  torch.save(state_dict, path)

# load the states
def load_AdaIN(path='AdaIN.pth', training_mode=False):
  # load the states
  state_dict = torch.load(path)
  # create a model, if there is encoder's state, do not used pretrained encoder from torchvision
  model = AdaIN(torchvision_encoder=('encoder' not in state_dict.keys()), training_mode=training_mode)
  try:
    # load the states of encoder
    model.encoder.load_state_dict(state_dict['encoder'])
  except KeyError:
    pass
  # load the states of decoder
  model.decoder.load_state_dict(state_dict['decoder'])

  # return loaded model
  return model
