"""
 This script simulates the evolution of an intially Gaussian wave packet 
 which hits a barrier. The barrier has a "smoothly rectangular" shape.
In addition to simulating the evolution of the wave packet, the script
estimates the transmission and reflection probabilities afther the 
 collision.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Tfinal  - The duration of the simulation
   dt      - The step size in time

 Inputs for the initial Gaussian:
   x0      - The mean position of the initial wave packet
   p0      - The mean momentum of the initial wave packet
   sigmaP  - The momentum width of the initial, Gaussian wave packet
   tau     - The time at which the Gaussian is narrowest (spatially)
 
 Input for the barrier:
   V0      - The height of the barrier (can be negative)
   w       - The width of the barrier
   s       - Smoothness parameter

 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider

# Numerical grid parameters
L = 300.0
N = 2048        # Should be 2**k, with k being an integer

# Simulation on/off boolean
running = True

# Numerical time parameters
Tfinal = 150;
dt = 0.5;

# Inputs for the smoothly rectangular potential
V0 = 2            # Heigth
s = 15.0             # Smoothness
width = 3.0         # Width

# Inputs for the absorbing potential
Gamma0 = 1e-4
Onset = 90

# Inputs for the Gaussian 
p0 = 1.0
sigmaX = 2;
sigmaP = 1/(2*sigmaX);
tau = 0.0

# Initial mean position
x0 = -10*sigmaX - width/2

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Window fixation values
xMin = -100
xMax = 100
yMin = -0.100
yMax = 0.200

# Determine double derivative by means of the fast Fourier transform.
# Set up vector of k-values
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Allocate and declare
Tmat_FFT = np.zeros((N, N), dtype=complex)
# Transform identity matrix
Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex))
# Multiply by (ik)^2
Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
# Transform back to x-representation. 
# Transpose necessary as we want to transform columnwise
Tmat_FFT = np.fft.ifft(np.transpose(Tmat_FFT))
# Correct pre-factor
Tmat_FFT = -1/2*Tmat_FFT    

# Add potential
Vpot = V0/(np.exp(s*(np.abs(x)-width/2))+1.0)
# Absorber
Gamma = (np.abs(x) > Onset)*Gamma0*(np.abs(x) - Onset)**2

# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot -1j*Gamma)
# Propagator
U = linalg.expm(-1j*Ham*dt)

# Set up Gaussian - analytically
InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
# Initial Gaussian
Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)

# Initiate plots
plt.ion()
fig = plt.figure(1, figsize=(10,8))
plt.clf()

# Initiate gridspecs
gs0 = gridspec.GridSpec(5,1, figure=fig)
gs00 = gs0[4].subgridspec(5,8)
gs000 = gs00[4, :8].subgridspec(1,8)

# Start function
def start(event):
    global t, Psi, Psi0, Vpot, Ham, U, V0, width, p0, sigmaX, sigmaP, line2, Psi0Max, running, s, x, tau, x0
    
    # Reset t
    t = 0

    # Set values from sliders
    V0 = ax2_slider.val
    width = ax3_slider.val
    p0 = ax4_slider.val
    sigmaX = ax5_slider.val
    sigmaP = 1/(2*sigmaX)

    # Add potential
    Vpot = V0/(np.exp(s*(np.abs(x)-width/2))+1.0)

    # Full Hamiltonian
    Ham = Tmat_FFT + np.diag(Vpot -1j*Gamma)
    U = linalg.expm(-1j*Ham*dt)

    x0 = -7*sigmaX - width/2

    # Set up Gaussian - analytically
    InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))

    # Initial Gaussian
    Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)
    
    Psi = Psi0
    Psi0Max = np.max(np.abs(Psi0)**2)

    # Redraw barrier y axis
    line2.set_ydata(V0 * Psi0Max/(np.exp(s*(np.abs(x)-width/2))+2.5))
    running = True

# Stop function
def stop(event):
    global running
    running = False

# Initiate axn
ax1 = fig.add_subplot(gs0[:4])

# Fix window
ax1.set(xlim=(xMin, xMax), ylim=(yMin, yMax))

# Plot line
line1, = ax1.plot(x, np.abs(Psi0)**2, '-', color='black', 
                 label = 'Analytical')

# Scaling and plotting the potential
Psi0Max = np.max(np.abs(Psi0)**2)
line2, = ax1.plot(x, V0 * Psi0Max/(np.exp(s*(np.abs(x)-width/2))+2.5), '-', 
                 color='red', label = 'FD3')

# Initiate sliders and buttons
#Barrier height slider
ax2 = fig.add_subplot(gs00[0, 1:7])
ax2_slider = Slider(
    ax=ax2,
    label='Barrier Height',
    valmin=-1,
    valmax=2,
    valinit=V0,
)

# Barrier width slider
ax3 = fig.add_subplot(gs00[1, 1:7])
ax3_slider = Slider(
    ax=ax3,
    label='Barrier Width',
    valmin = 1,
    valmax = 5,
    valinit = width,
)

# Gaussian velocity slider
ax4 = fig.add_subplot(gs00[2, 1:7])
ax4_slider = Slider(
    ax=ax4,
    label='Initial Gaussian Velocity',
    valmin= .5,
    valmax= 2,
    valinit = p0,
)

# Gaussian width slider
ax5 = fig.add_subplot(gs00[3, 1:7])
ax5_slider = Slider(
    ax=ax5,
    label='Initial Gaussian Width',
    valmin = 1,
    valmax = 10,
    valinit = sigmaX,
)

# Start button
ax6 = fig.add_subplot(gs000[0,1])
startButton = Button(ax6, 'Restart', hovercolor='0.975')
startButton.on_clicked(start)

# Stop button
ax7 = fig.add_subplot(gs000[0,6])
stopButton = Button(ax7, 'Stop', hovercolor='0.975')
stopButton.on_clicked(stop)


# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi = Psi0
t = 0

while True:
  if (t < (np.abs(x0) / p0) or Integral > 1e-4) and running:
    # Update time
    t = t+dt              
    Psi = np.matmul(U, Psi)
    # Update data for plots
    line1.set_ydata(np.power(np.abs(Psi), 2))
    Integral = np.trapz(np.power(np.abs(np.asarray(Psi.T)), 2)*(np.abs(x) < 2 * width), dx = h)
  # Update plots
  fig.canvas.draw()
  fig.canvas.flush_events()