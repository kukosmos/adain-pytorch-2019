assert __name__ == '__main__', 'This file cannot be imported.'

import argparse

parser = argparse.ArgumentParser(description='AdaIN Training Script')

# necessary arguments
parser.add_argument('-cd', '--content-dir', type=str, metavar='<dir>', required=True, help='Directory with content images')
parser.add_argument('-sd', '--style-dir', type=str, metavar='<dir>', required=True, help='Directory with style images')

# optional arguments for training
parser.add_argument('--save-dir', type=str, metavar='<dir>', default='./experiments', help='Directory to save trained models, default=./experiments')
parser.add_argument('--log-dir', type=str, metavar='<dir>', default='./logs', help='Directory to save logs, default=./logs')
parser.add_argument('--log-image-every', type=int, metavar='<int>', default=100, help='Period for loging generated images, non-positive for disabling, default=100')
parser.add_argument('--save-interval', type=int, metavar='<int>', default=10000, help='Period for saving model, default=10000')
parser.add_argument('--include-encoder', action='store_true', help='Option for saving with the encoder')
parser.add_argument('--cuda', action='store_true', help='Option for using GPU if available')
parser.add_argument('--n-threads', type=int, metavar='<int>', default=2, help='Number of threads used for dataloader, default=2')

# hyper-parameters
parser.add_argument('--learning-rate', type=float, metavar='<float>', default=1e-4, help='Learning rate, default=1e-4')
parser.add_argument('--learning-rate-decay', type=float, metavar='<float>', default=5e-5, help='Learning rate decay, default=5e-5')
parser.add_argument('--max-iter', type=int, metavar='<int>', default=160000, help='Maximun number of iteration, default=160000')
parser.add_argument('--batch-size', type=int, metavar='<int>', default=8, help='Size of the batch, default=8')
parser.add_argument('--style-weight', type=float, metavar='<float>', default=10.0, help='Weight of style loss, default=10.0')
parser.add_argument('--content-weight', type=float, metavar='<float>', default=1.0, help='Weight of content loss, default=1.0')

args = parser.parse_args()

import os
import torch
import torch.utils.data as data

from dataloader import ImageFolderDataset, InfiniteSampler, train_transform
from network import AdaIN, save_AdaIn
from pathlib import Path
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import learning_rate_decay

# for handling errors
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# use gpu if possible
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

# directory trained models
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
# directory for logs
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)

# content dataset
content_dataset = ImageFolderDataset(args.content_dir, train_transform((512, 512), 256))
content_iter = iter(data.DataLoader(content_dataset, batch_size=args.batch_size, sampler=InfiniteSampler(len(content_dataset)), num_workers=args.n_threads))
# style dataset
style_dataset = ImageFolderDataset(args.style_dir, train_transform((512, 512), 256))
style_iter = iter(data.DataLoader(style_dataset, batch_size=args.batch_size, sampler=InfiniteSampler(len(style_dataset)), num_workers=args.n_threads))

# AdaIN model
model = AdaIN()
model.to(device)
optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.learning_rate)

# log writer
writer = SummaryWriter(log_dir=str(log_dir))

# for maximum iteration
for i in tqdm(range(args.max_iter)):
  # adjust learning rate
  lr = learning_rate_decay(args.learning_rate, args.learning_rate_decay, i)
  for group in optimizer.param_groups:
    group['lr'] = lr

  # get images
  content_images = next(content_iter).to(device)
  style_images = next(style_iter).to(device)

  # calculate loss
  g, loss_content, loss_style = model(content_images, style_images)
  loss_content = args.content_weight * loss_content
  loss_style = args.style_weight * loss_style
  loss = loss_content + loss_style

  # optimize the network
  optimizer.zero_grad()  
  loss.backward()
  optimizer.step()

  # write logs
  writer.add_scalar('Loss/Loss', loss.item(), i + 1)
  writer.add_scalar('Loss/Loss_content', loss_content.item(), i + 1)
  writer.add_scalar('Loss/Loss_style', loss_style.item(), i + 1)
  if args.log_image_every > 0 and ((i + 1) % args.log_image_every == 0 or i == 0 or (i + 1) == args.max_iter):
    writer.add_image('Image/Content', content_images[0], i + 1)
    writer.add_image('Image/Style', style_images[0], i + 1)
    writer.add_image('Image/Generated', g[0], i + 1)

  # save model
  if (i + 1) % args.save_interval == 0 or (i + 1) == args.max_iter:
    save_AdaIn(model, os.path.join(save_dir, 'iter_{}.pth'.format(i + 1)), include_encoder=args.include_encoder)

writer.close()
