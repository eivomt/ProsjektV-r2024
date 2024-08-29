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
from matplotlib.widgets import Button, Slider
import matplotlib.gridspec as gridspec

# Layout
plt.rcParams['figure.constrained_layout.use'] = True

# Numerical grid parameters
L = 400.0
N = 1024         # Should be 2**k, with k being an integer
Integral = 0

# Numerical time parameters
# Tfinal = 100
dt = 0.5

# Inputs for the smoothly rectangular potential
V0 = 1.0            # Heigth
s = 15.0             # Smoothness
width = 2.0         # Width

# Inputs for the Gaussian 
x0 = -20.0
p0 = 1.0
sigmaP = 0.2
tau = 0.0


# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Determine double derivative by means of the fast Fourier transform.
# Set up vector of k-values
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Allocate and declare
Tmat_FFT = np.zeros((N, N), dtype=complex)
# Transform identity matrix
Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
# Transform back to x-representation. 
Tmat_FFT = np.fft.ifft(Tmat_FFT, axis = 0)
# Correct pre-factor
Tmat_FFT = -1/2*Tmat_FFT    

# Add potential
Vpot = V0/(np.exp(s*(np.abs(x)-width/2))+1.0)
# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot)
# Propagator
U = linalg.expm(-1j*Ham*dt)

# Set up Gaussian - analytically
InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
# Initial Gaussian
Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)

# Initiate plots
plt.ion()
# fig = plt.figure(1, layout='constrained')
fig = plt.figure(constrained_layout=True)
spec = fig.add_gridspec(4,1)
ax = fig.add_subplot(spec[0,0])
# fig = plt.figure(1)
# ax = fig.add_subplot()
ax.set(ylim=(-.1, .2))                # Fix window
ax.set(xlim=(-100, 100))                # Fix window
line1, = ax.plot(x, np.abs(Psi0)**2, '-', color='black')
# Scaling and plotting the potential
Psi0Max = np.max(np.abs(Psi0)**2)
Faktor = 1
line2, = ax.plot(x, Faktor*Vpot, '-', color='red')

# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi = Psi0
t = 0


fig.set_size_inches(6.4, 8.0)
# fig.subplots_adjust(bottom=0.25)

axbheight = fig.add_subplot(spec[1,0])
# axbheight = fig.add_axes([0.25, 0.01, 0.65, 0.03])
# axbheight = fig.add_axes(111)
barrier_height_slider = Slider(
    ax=axbheight,
    label='Barrier Height',
    valmin=-2,
    valmax=2,
    valinit=V0,
)

axbheight2 = fig.add_subplot(spec[2,0])
barrier_height_slider2 = Slider(
    ax=axbheight2,
    label='Barrier Height',
    valmin=-2,
    valmax=2,
    valinit=V0,
)

resetax = fig.add_subplot(spec[3,0])
button = Button(resetax, 'Reset', hovercolor='0.975')

print(fig.get_size_inches())

def reset(event):
    global t, Psi, V0, Vpot, Ham, U
    Psi = Psi0
    t = 0
    V0 = barrier_height_slider.val
    # Add potential
    Vpot = V0/(np.exp(s*(np.abs(x)-width/2))+1.0)
    # Full Hamiltonian
    Ham = Tmat_FFT + np.diag(Vpot)
    U = linalg.expm(-1j*Ham*dt)
    print(barrier_height_slider.val)
button.on_clicked(reset)

# Iterate while time t is less than final time Tfinal
while True:
  line2.set_ydata(Faktor*Vpot)
  if t < (np.abs(x0) / p0) or Integral > 1e-4:
    # Update time
    t = t+dt              
    Psi = np.matmul(U, Psi)
    # Update data for plots
  
    line1.set_ydata(np.power(np.abs(Psi), 2))
    Integral = np.trapz(np.power(np.abs(np.asarray(Psi.T)), 2)*(np.abs(x) < 2 * width), dx = h)
  # Update plots
  fig.canvas.draw()
  fig.canvas.flush_events()


# Estimate reflection and transmission probabilities
# For reflection probability we multiply Psi with 0 for x<0,
# for reflection probability we multiply Psi with 0 for x>0
#
# Change format of arrays
Psi = np.asarray(Psi.T)
R = np.trapz((x < 0)*np.abs(Psi)**2, dx = h)
T = np.trapz((x > 0)*np.abs(Psi)**2, dx = h)
# Print result to screen
print('Transmission probability: ', float(100*T), '%')
print('Reflection probability: ', float(100*R), '%')
