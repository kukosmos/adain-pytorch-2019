assert __name__ == '__main__'

import argparse

parser = argparse.ArgumentParser(description='AdaIN Training Script')

parser.add_argument('-cd', '--content-dir', type=str, metavar='<dir>', required=True, help='Directory with content images')
parser.add_argument('-sd', '--style-dir', type=str, metavar='<dir>', required=True, help='Directory with style images')

parser.add_argument('--save-dir', type=str, metavar='<dir>', default='./experiments', help='Directory to save trained models, default=./experiments')
parser.add_argument('--log-dir', type=str, metavar='<dir>', default='./logs', help='Directory to save logs, default=./logs')
parser.add_argument('--save-interval', type=int, metavar='<int>', default=10000, help='Period for saving model, default=10000')
parser.add_argument('--include-encoder', action='store_true', help='Option for saving with the encoder')
parser.add_argument('--cuda', action='store_true', help='Option for using GPU if available')
parser.add_argument('--n-threads', type=int, metavar='<int>', default=2, help='Number of threads used for dataloader, default=2')

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
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import learning_rate_decay

device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)

content_dataset = ImageFolderDataset(args.content_dir, train_transform((512, 512), 256))
content_iter = iter(data.DataLoader(content_dataset, batch_size=args.batch_size, sampler=InfiniteSampler(len(content_dataset)), num_workers=args.n_threads))
style_dataset = ImageFolderDataset(args.style_dir, train_transform((512, 512), 256))
style_iter = iter(data.DataLoader(style_dataset, batch_size=args.batch_size, sampler=InfiniteSampler(len(style_dataset)), num_workers=args.n_threads))

model = AdaIN()
model.to(device)
optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.learing_rate)

writer = SummaryWriter(log_dir=str(log_dir))

for i in tqdm(range(args.max_iter)):
  lr = learning_rate_decay(args.learning_rate, args.learning_rate_decay, i)
  for group in optimizer.param_groups:
    group['lr'] = lr
  content_images = next(content_iter).to(device)
  
  style_images = next(style_iter).to(device)
  content_images = next(content_iter).to(device)

  _, loss_content, loss_style = model(content_images, style_images)
  loss_content = args.content_weight * loss_content
  loss_style = args.style_weight * loss_style
  loss = loss_content + loss_style

  optimizer.zero_grad()  
  loss.backward()
  optimizer.step()

  writer.add_scalar('loss_content', loss_content.item(), i + 1)
  writer.add_scalar('loss_style', loss_style.item(), i + 1)

  if (i + 1) % args.save_interval == 0 or (i + 1) == args.max_iter:
    save_AdaIn(model, os.path.join(save_dir, 'iter_{}.pth'.format(i + 1)), include_encoder=args.include_encoder)

writer.close()
