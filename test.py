assert __name__ == '__main__', 'This file cannot be imported'

import argparse

parser = argparse.ArgumentParser(description='AdaIN Testing Script')

# necessary arguments
# content images or folder that contains content images, mutually exclusive
content_group = parser.add_mutually_exclusive_group(required=True)
content_group.add_argument('-c', '--content', type=str, metavar='<file>', nargs='+', help='Content image(s)')
content_group.add_argument('-cd', '--content-dir', type=str, metavar='<dir>', help='Directory with content images')
# style images or folder that contains style images, mutually exclusive
style_group = parser.add_mutually_exclusive_group(required=True)
style_group.add_argument('-s', '--style', type=str, metavar='<file>', nargs='+', help='Style image(s)')
style_group.add_argument('-sd', '--style-dir', type=str, metavar='<dir>', help='Directory with style images')
parser.add_argument('-m', '--model', type=str, required=True, metavar='<pth>', help='Trained AdaIN model')

# optional arguments for generating images
parser.add_argument('--cuda', action='store_true', help='Option for using GPU if available')
parser.add_argument('--content-size', type=int, metavar='<int>', default=512, help='Size for resizing content images, 0 for keeping original size, default=512')
parser.add_argument('--style-size', type=int, metavar='<int>', default=512, help='Size for resizing style images, 0 for keeping original size, default=512')
parser.add_argument('--crop', action='store_true', help='Option for central crop')
parser.add_argument('--ext', type=str, metavar='<ext>', default='.jpg', help='Extension name of the created images, default=.jpg')
parser.add_argument('--output', type=str, metavar='<dir>', default='./results', help='Directory for saving created images, default=./results')

# advanced options
parser.add_argument('--alpha', type=float, metavar='<float>', default=1.0, help='Option for degree of stylization, should be between 0 and 1, default=1.0')
advanced_group = parser.add_mutually_exclusive_group()
advanced_group.add_argument('--preserve-color', action='store_true', help='Option for preserving color of created images')
advanced_group.add_argument('--interpolation-weights', type=int, metavar='<int>', nargs='+', help='Weights of style images for interpolation')

args = parser.parse_args()

import torch

from dataloader import test_transform
from network import load_AdaIN
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
from utils import color_control

# use gpu if possible
device = torch.device('cuda' if args.cuda and torch.cuda.is_avaiable() else 'cpu')

# directory for save generated images
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# path to content images
if args.content:
  contents = [Path(c) for c in args.content]
else:
  content_dir = Path(args.content_dir)
  contents = list(content_dir.glob('**/*.*'))

# path to style images
if args.style:
  styles = [Path(s) for s in args.style]
else:
  style_dir = Path(args.style_dir)
  styles = list(style_dir.glob('**/*.*'))

# if interpolation weights are presented, mix the styles with weights
if args.interpolation_weights:
  assert len(styles) == len(args.interpolation_weights), 'All style images should be weighted, {} images are given while {} weights are given'.format(len(styles), len(args.interpolation_weights))
  interpolation = True
  sum_weights = sum(args.interpolation_weights)
  interpolation_weights = [w / sum_weights for w in args.interpolation_weights]
  interpolation_weights = torch.tensor(interpolation_weights)
  interpolation_weights = interpolation_weights.unsqueeze(0).to(device)
else:
  interpolation = False

# data handle helper
content_transform = test_transform(args.content_size, args.crop)
style_transform = test_transform(args.style_size, args.crop)

# import trained model
model = load_AdaIN(args.model, training_mode=False)
model.to(device)

# for all content images
for content_path in contents:
  # if interpolation weights are presented
  if interpolation:
    print('Content: {}, Style: interpolation'.format(content_path))

    # get style images and stack up style images as a batch
    style = torch.stack([style_transform(Image.open(str(p))) for p in styles]).to(device)
    # get a content image
    content = content_transform(Image.open(str(content_path))).unsqueeze(0).to(device)

    # generate a styled image with interpolated styles
    with torch.no_grad():
      output = model(content, style, interpolation_weights=interpolation_weights)
    output = output.cpu()

    # save the generated image
    save_image(output, str(output_dir / '{}_interpolation{}'.format(content_path.stem, args.ext)))

  # otherwise, if interpolation weight are not presented
  else:
    # for all style images
    for style_path in styles:
      print('Content: {}, Style: {}'.format(content_path, style_path))

      # get a single content image
      content = content_transform(Image.open(str(content_path)))
      content = content.unsqueeze(0).to(device)

      # get a single style image
      style = style_transform(Image.open(str(style_path))).to(device)
      # change the color of style image into the color of content image
      if args.preserve_color:
        style = color_control(style, content)
      style = style.to(device).unsqueeze(0)
      
      # generate a styled image
      with torch.no_grad():
        output = model(content, style, alpha=args.alpha)
      output = output.cpu()

      # save the generated image
      save_image(output, str(output_dir / '{}_stylized_{}{}'.format(content_path.stem, style_path.stem, args.ext)))
