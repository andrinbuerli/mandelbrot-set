import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define a higher resolution for plotting the Mandelbrot set
width_high_res, height_high_res = 1000, 1000  # 10-fold resolution increase
xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5

# zoomed 
#xmin, xmax, ymin, ymax = -1.0, -0.5, -0.5, 0.0

max_iter_high_res = 100

# Compute the Mandelbrot set at higher resolution
def mandelbrot_set_high_res(xmin, xmax, ymin, ymax, width, height, max_iter, return_mag=False):
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
        z[mask] = 0  # Avoid further computations for diverged points
        z_magnitude = np.where(divergence_step == 0, abs(z), 2 * divergence_step)

    return divergence_step if not return_mag else z_magnitude


return_mag = True
# Generate the Mandelbrot set data
mandelbrot_high_res = mandelbrot_set_high_res(
    xmin, xmax, ymin, ymax, width_high_res, height_high_res, max_iter_high_res, return_mag=return_mag
)

# Plot the high-resolution Mandelbrot set
plt.figure(figsize=(12, 12), dpi=300)
plt.imshow(
    mandelbrot_high_res.T,
    extent=[xmin, xmax, ymin, ymax],
    cmap="hot",
    origin="lower",
)
plt.colorbar(label="Iterations to Diverge" if not return_mag else "Final Magnitude")
plt.title("Mandelbrot Set")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.savefig("mandelbrot_high_res.png")
#plt.savefig("mandelbrot_high_res_zoomed.png")
