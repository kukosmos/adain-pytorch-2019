def calc_mean_std(feat, eps=1e-5):
  size = feat.size()
  N, C = size[:2]

  feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

  feat_var = feat.view(N, C, -1).var(dim=2) + eps
  feat_std = feat_var.sqrt().view(N, C, 1, 1)

  return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
  size = content_feat.size()
  
  style_mean, style_std = calc_mean_std(style_feat)
  content_mean, content_std = calc_mean_std(content_feat)
  
  normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

  return normalized_feat * style_std.expand(size) + style_mean.expand(size)
