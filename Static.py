"""
This script simulates the relationship between knowledge of the position of a particle 
and knowledge of its velocity. It aims to illustrate the principle of uncertainty,
in which the more certain one is on the particles velocity, ones uncertainty as pertains to 
its position increases, proportional to that very certainty. 
What a strange and wonderful world we live in.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points

 Inputs for the initial Gaussian:
   x0      - The mean position of the wave packet
   p0      - The mean momentum of the wave packet
   sigmaP  - The momentum width of the wave packet
 
 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# Numerical grid parameters
Lx = 50
Lp = 20
N = 1001        # Should be 2**k, with k being an integer

# Inputs for the Gaussian 
p0 = 0
sigmaX = 2
sigmaP = 1/(2*sigmaX)

# Initial mean position
x0 = 0

# Set up grid
x = np.linspace(-Lx/2, Lx/2, N)
p = np.linspace(-Lp / 2, Lp / 2, N)

# Window fixation values
xMin = -Lx / 2
xMax = Lx / 2
yMin = 0
yMax = 1.00

xpMin = -Lp / 2
xpMax = Lp / 2
ypMin = 0
ypMax = 5.00

# Initiate plots
plt.ion()
fig = plt.figure(1, figsize=(10,8))
plt.clf()

# Initiate gridspecs
gs0 = gridspec.GridSpec(7,1, figure=fig)
gs00 = gs0[6].subgridspec(3,1)

# Initiate axn
ax1 = fig.add_subplot(gs0[0:2])
ax2 = fig.add_subplot(gs0[3:5])

# Fix windows
ax1.set(xlim=(xMin, xMax), ylim=(yMin, yMax))
ax2.set(xlim=(xpMin, xpMax), ylim=(ypMin, ypMax))

# Plot lines
line1, = ax1.plot(x, (1/(sigmaX * np.sqrt(2 * np.pi))) * (np.exp(-(x - x0)**2 / (2 * sigmaX**2))), '-', color='black', label = 'Analytical')
line2, = ax2.plot(p, (1/(sigmaP * np.sqrt(2 * np.pi))) * np.exp(-(p - p0)**2 / (2 * sigmaP**2)), '-', color='black', label = 'Analytical')

# Set titles
ax1.title.set_text('Posisjon-fordeling')
ax2.title.set_text('Momentum-fordeling')

# Update function
def update(event):
    # import global variables
    global line1, line2, x, p0, x0, sigmaX, sigmaP, p

    # set new values
    x0 = ax3_slider.val
    p0 = ax4_slider.val
    sigmaX = ax5_slider.val
    sigmaP = 1/(2*sigmaX)

    # update y axis data
    line1.set_ydata((1/(sigmaX * np.sqrt(2 * np.pi))) * (np.exp(-(x - x0)**2 / (2 * sigmaX**2))))
    line2.set_ydata((1/(sigmaP * np.sqrt(2 * np.pi))) * (np.exp(-(p - p0)**2 / (2 * sigmaP**2))))

# Initiate sliders
ax3 = fig.add_subplot(gs00[0])
ax3_slider = Slider(
    ax=ax3,
    label='$x_0$',
    valmin=xMin,
    valmax=xMax,
    valinit=x0
)

ax4 = fig.add_subplot(gs00[1])
ax4_slider = Slider(
    ax=ax4,
    label='$p_0$',
    valmin=xpMin,
    valmax=xpMax,
    valinit=p0
)

ax5 = fig.add_subplot(gs00[2])
ax5_slider = Slider(
    ax=ax5,
    label=r'$\sigma_x$',
    valmin=.5,
    valmax=5,
    valinit=sigmaX
)

# Connect sliders to update function
ax3_slider.on_changed(update)
ax4_slider.on_changed(update)
ax5_slider.on_changed(update)

# Infinite draw loop
while True:
    fig.canvas.draw()
    fig.canvas.flush_events()