assert __name__ == '__main__'

import argparse

parser = argparse.ArgumentParser(description='AdaIN Testing Script')

parser.add_argument('-c', '--content', type=str, required=True, metavar='<dir>', help='Content directory(or file)')
parser.add_argument('-s', '--style', type=str, required=True, metavar='<dir>', help='Style directory(or file)')
parser.add_argument('-m', '--model', type=str, required=True, metavar='<pth>', help='Trained AdaIN model')

parser.add_argument('--content-size', type=int, metavar='<int>', default=512, help='Size for resizing content images, 0 for keeping original size, default=512')
parser.add_argument('--style-size', type=int, metavar='<int>', default=512, help='Size for resizing style images, 0 for keeping original size, default=512')
parser.add_argument('--crop', action='store_true', help='Option for central crop')

parser.add_argument('--ext', type=str, metavar='<ext>', default='.jpg', help='Extension name of the created images, default=.jpg')
parser.add_argument('--output', type=str, metavar='<dir>', default='./results', help='Directory for saving created images, default=./results')

parser.add_argument('--preserve-color', action='store_true', help='Option for preserving color of created images')
parser.add_argument('--alpha', type=float, metavar='<float>', default=1.0, help='Option for degree of stylization, should be between 0 and 1, default=1.0')
parser.add_argument('--interpolation-weights', type=int, metavar='<int>', nargs='+', help='Weights of style images for interpolation')

args = parser.parse_args()

from network import AdaIN
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image


