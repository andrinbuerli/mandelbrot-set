import argparse
import torch
from tqdm import tqdm
from mbsnn.mandelbrot_setnn import MandelbrotNN

parser = argparse.ArgumentParser(description='Render Mandelbrot Set with a trained model')
parser.add_argument('--checkpoint', default="", help='Path to the checkpoint')
parser.add_argument('--xmin', default=-2.0, help='Minimum x value')
parser.add_argument('--xmax', default=1.0, help='Maximum x value')
parser.add_argument('--ymin', default=-1.5, help='Minimum y value')
parser.add_argument('--ymax', default=1.5, help='Maximum y value')
parser.add_argument('--width', default=512, help='Width of the image')
parser.add_argument('--height', default=512, help='Height of the image')
parser.add_argument('--iterations', nargs='+', default=[5,10,20,40], help='List of iterations')
parser.add_argument('--zoomed', action='store_true', help='Zoomed in image')
args = parser.parse_args()

nn = MandelbrotNN(num_hidden_layers=6, hidden_dim=2048, log_scale=False)
nn.load_state_dict(torch.load(args.checkpoint)["state_dict"])

zoom_args = dict(real_min=args.xmin, real_max=args.xmax, imag_min=args.ymin, imag_max=args.ymax)
    
if args.zoomed:
    zoom_args = dict(real_min=-1.0, real_max=-0.5, imag_min=-0.5, imag_max=0.0)
    
for iter in tqdm(args.iterations):
    iter = int(iter)
    fig = nn.plot_mandelbrot_set(iterations=iter, **zoom_args)
    
    fig.tight_layout()
    fig.savefig(f"mandelbrot_nn_{iter}.png", dpi=300)