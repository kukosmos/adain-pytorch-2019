import torch

def calc_mean_std(feat, eps=1e-5):
  size = feat.size()
  N, C = size[:2]

  feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

  feat_var = feat.view(N, C, -1).var(dim=2) + eps
  feat_std = feat_var.sqrt().view(N, C, 1, 1)

  return feat_mean, feat_std

# presented in "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization, https://arxiv.org/abs/1703.06868"
def adaptive_instance_normalization(content_feat, style_feat):
  size = content_feat.size()
  
  style_mean, style_std = calc_mean_std(style_feat)
  content_mean, content_std = calc_mean_std(content_feat)
  
  normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

  return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def learning_rate_decay(lr, decay, iteration):
  return lr / (1.0 + decay * iteration)

def calc_flatten_mean_std(feat):
  flatten = feat.view(3, -1)
  mean = flatten.mean(dim=-1, keepdim=True)
  std = flatten.std(dim=-1, keepdim=True)
  return flatten, mean, std

def matrix_sqrt(mat):
  U, D, V = torch.svd(mat)
  return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

# presented in "Controlling Perceptual Factors in Neural Style Transfer, https://arxiv.org/abs/1611.07865"
def color_control(original, target):
  flatten_o, mean_o, std_o = calc_flatten_mean_std(original)
  normalized_o = (flatten_o - mean_o.expand_as(flatten_o)) / std_o.expand_as(flatten_o)
  cov_eye_o = torch.mm(normalized_o, normalized_o.t()) + torch.eye(3)

  flatten_t, mean_t, std_t = calc_flatten_mean_std(target)
  normalized_t = (flatten_t - mean_o.expand_as(flatten_t)) / std_o.expand_as(flatten_t)
  cov_eye_t = torch.mm(normalized_t, normalized_t.t()) + torch.eye(3)

  normalized_transfer = torch.mm(matrix_sqrt(cov_eye_t), torch.mm(torch.inverse(matrix_sqrt(cov_eye_o)), normalized_o))
  original_transfer = normalized_transfer * std_t.expand_as(normalized_o) + mean_t.expand_as(normalized_o)

  return original_transfer.view(original.size())
