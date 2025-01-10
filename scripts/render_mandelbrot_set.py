import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# add argparse
import argparse
cli = argparse.ArgumentParser(description="Render the Mandelbrot set")
cli.add_argument("--width", type=int, default=1000, help="Width of the output image")
cli.add_argument("--height", type=int, default=1000, help="Height of the output image")
cli.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
cli.add_argument("--xmin", type=float, default=-2, help="Minimum x-value")
cli.add_argument("--xmax", type=float, default=1, help="Maximum x-value")
cli.add_argument("--ymin", type=float, default=-1.5, help="Minimum y-value")
cli.add_argument("--ymax", type=float, default=1.5, help="Maximum y-value")
cli.add_argument("--output", type=str, default="mandelbrot.png", help="Output file name")
cli.add_argument("--zoomed", action="store_true", help="zoomed in")
cli.add_argument("--render_mag", action="store_true", help="render magnitude of the final z value")
args = cli.parse_args()

# Define a higher resolution for plotting the Mandelbrot set
width_high_res, height_high_res = args.width, args.height
xmin, xmax, ymin, ymax = args.xmin, args.xmax, args.ymin, args.ymax

if args.zoomed:
    # zoomed in
    xmin, xmax, ymin, ymax = -1.0, -0.5, -0.5, 0.0

# Compute the Mandelbrot set at higher resolution
def mandelbrot_set_high_res(xmin, xmax, ymin, ymax, width, height, max_iter, render_mag=False):
    # Create a grid of complex numbers
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    c = real[:, np.newaxis] + 1j * imag[np.newaxis, :]
    z = np.zeros_like(c, dtype=np.complex128)
    divergence_step = np.zeros(c.shape, dtype=int)  # Tracks iterations at divergence

    for i in tqdm(list(range(max_iter))):
        z = z**2 + c
        mask = (abs(z) > 2) & (divergence_step == 0)
        divergence_step[mask] = i
        #z[mask] = 0  # Avoid further computations for diverged points

    return divergence_step if not render_mag else abs(z)

# Generate the Mandelbrot set data
mandelbrot_high_res = mandelbrot_set_high_res(
    xmin, xmax, ymin, ymax, width_high_res, height_high_res, args.max_iter, render_mag=args.render_mag
)

# Plot the high-resolution Mandelbrot set
plt.figure(figsize=(12, 12), dpi=300)
plt.imshow(
    np.where(((~np.isnan(mandelbrot_high_res.T)) & (mandelbrot_high_res.T < 2.0)), mandelbrot_high_res.T, 2.0)
    if args.render_mag else mandelbrot_high_res.T,
    extent=[xmin, xmax, ymin, ymax],
    cmap="hot",
    origin="lower",
    interpolation="none",
)
plt.colorbar(label="Iterations to Diverge" if not args.render_mag else "Clipped Magnitude")
plt.title(f"Mandelbrot Set after {args.max_iter} Iterations")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.savefig(args.output)
#plt.savefig("mandelbrot_high_res_zoomed.png")
