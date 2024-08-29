"""
 This script simulates the time-evolution of the wave packet for a 
 particle trapped in a confining potential. The initial state is fixed by
 randomly selecting the coefficients in a linear combination of the 
 eigenstates of the Hamiltonian.

 The evolution of the wave function is plotted until some final time at
 which an energy mesurement is done. After this, the wave function is
 collapsed to one of the eigenstates. The state is picked at random
 according to the probability distribution given by the initial
 coefficients.

 Numerical input parameters: 
 Tmeasure  - the duration of the simulation - until the measurement
 dt        - numerical time step, here it serves to tune the speed of the
 simulation
 N         - number of grid points, should be 2^n
 L         - the size of the numerical domain it extends from -L/2 to L/2
 
 Physical input parameters:
 V0        - the depth of the confinint potential (must be negative)
 s         - smoothness parameter for the potential
 w         - the width of the smoothly rectangular potential
 Avector   - vector with coefficients for the initial state
 
 All these inputs are hard coded initially.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt

# Numerical time parameters:
Tmeasure = 10
dt = 0.1

# Grid parameters
L = 30
N = 512              # For FFT's sake, we should have N=2^n

# Physical parameters:
V0 = -4
s = 5
w = 6

# Vector with coefficients - we assign 10. In case there are more 
# bound states than this, this should be augmented
Avector = np.zeros(N, dtype = complex)
Avector[0:10] = np.array([2, 3-0.2j, 1j, 1.5+0.5j, 2, 1.4, 1-1j, 1-1j, -2, .2-1j])

# Potential
def Vpot(x):
    return V0/(np.exp(s*(np.abs(x)-w/2))+1)


# Set up the grid.
x = np.linspace(-L/2, L/2, N)
h = L/(N+1)

# Set up Hamiltonian
# Kinetic energy:
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Transform identity matrix
Tmat = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat = np.matmul(np.diag(-k**2), Tmat)
# Transform back to x-representation. 
Tmat = np.fft.ifft(Tmat, axis = 0)
# Correct pre-factor
Tmat = -1/2*Tmat    

# Add kinetic and potential energy:
Ham = Tmat + np.diag(Vpot(x))

# Diagaonalize Hamiltonian (Hermitian matrix)
EigValues, EigVectors = np.linalg.eigh(Ham)
# Normalize eigenstates
EigVectors = EigVectors/np.sqrt(h)

# Number of bound states
Nbound = sum(EigValues < 0)
print(f'The potential supports {Nbound} bound states.')

    # Remove unbound states
EigValues = EigValues[0:Nbound]
EigVectors = EigVectors[:, 0:Nbound]

#
# Truncate vector with amplitudes and normalize it
Norm = np.sqrt(sum(np.abs(Avector[0:Nbound])**2))    
Avector = np.array(Avector[0:Nbound])/Norm

#
# Wave function, linear combination of eigenstates with coefficients given
# in Avector
Psi = np.matmul(EigVectors, Avector.T)

# Initiate time
t=0

# Initiate plots
fig, (ax1, ax2) = plt.subplots(1, 2, num=1)
# First plot: Histogram with probabilities
ax1.cla()
ax1.bar(np.arange(1, Nbound+1), np.abs(Avector)**2, color = 'red'
        )
ax1.set(ylim = (0, 1.1))
ax1.set_xlabel('n', fontsize = 12)
ax1.set_ylabel('$|a_n|^2$', fontsize = 12)

# Second plot: Wave packet and potential
ax2.cla()
ax2.plot(x, Vpot(x), '-', color = 'blue', linewidth = 2)
ax2.hlines(EigValues, -1.5*w, 1.5*w, linestyles = 'dashed', color = 'red')    
MaxVal = np.max(np.abs(Psi)**2)
line2, = ax2.plot(x, np.abs(Psi)**2/MaxVal*abs(V0) + V0, '-', color='black', 
                 linewidth = 2)
plt.xlabel('Position, x', fontsize = 12)
plt.ylabel(r'$|\Psi(x; t)|^2$', fontsize = 12)
ax2.set(xlim = (-1.5*w, 1.5*w), ylim=(1.2*V0, abs(V0)))                # Fix window

#
# Evolve in time
#
while t < Tmeasure:
  # Update time
  t = t + dt

  # Propagation
  AwithPhase = Avector*np.exp(-1j*EigValues*t)
  Psi = np.matmul(EigVectors, AwithPhase.T)

  # Update data for plots
  line2.set_ydata(np.abs(Psi)**2/MaxVal*abs(V0) + V0)
  # Update plots
  fig.canvas.draw()
  #fig.canvas.flush_events()
  plt.pause(0.01)
  

# Measurement
RandomNumber = np.random.rand()
DecisionVector = np.cumsum(np.abs(Avector)**2)
# Select eigenstate according to probability distribution
Ndraw = np.argmax(RandomNumber < DecisionVector)
# Write outcome to screen:
print(f'Outcome: State no. {Ndraw+1}, Energy: {EigValues[Ndraw]:.3f}.')
# Re-assign amplitudes and collapse wave function
Avector = np.zeros(Nbound)
Avector[Ndraw] = 1
# Update histogram
ax1.cla()
ax1.bar(np.arange(1, Nbound+1), np.abs(Avector)**2, color = 'red')
ax1.set_xlabel('n', fontsize = 12)
ax1.set_ylabel('$|a_n|^2$', fontsize = 12)

# Update plot of wave function
Psi = EigVectors[:, Ndraw]
line2.set_ydata(np.abs(Psi)**2/MaxVal*abs(V0) + EigValues[Ndraw])
# Update plots
fig.canvas.draw()
plt.show()