import numpy as np
import matplotlib.pyplot as plt

# Inputs for the well
V0 = -4            # Depth (negative)
w = 10             # Width
s = 5              # Smoothness

# Additional inputs for numerical approach
L = 40             # Size of grid
N = 1024           # Number of grid points

# Time inputs
Tmax = 10
dt = 0.025

# Amplitudes
AmplVector = np.array([1, 4, -2, 3, 0.5, 2, 1, 0.5, 0.2, 0.1])
AmplVector = AmplVector / np.sqrt(np.sum(np.abs(AmplVector)**2))  # Normalize
Nstates = len(AmplVector)

# Set up kinetic energy: FFT
x = np.linspace(-0.5, 0.5, N+2) * L
x = x[1:N+1]
h = x[1] - x[0]  # Spatial step size
wavenumFFT = 2 * np.pi / L * 1j * np.concatenate((np.arange(0, N/2), np.arange(-N/2, 0)))
T = -0.5 * np.fft.ifft(np.diag(wavenumFFT**2) @ np.fft.fft(np.eye(N)))

# Potential: Rectangular well
V = V0 / (np.exp(s * (np.abs(x) - w/2)) + 1)

# Hamiltonian
H = T + np.diag(V)

# Diagonalization
Eeig, U = np.linalg.eig(H)
Eeig = np.real(Eeig)
idx = Eeig.argsort()
Eeig = Eeig[idx]
U = U[:, idx]

# Normalize
U /= np.sqrt(h)

WF = np.zeros(N, dtype=complex)
for n in range(Nstates):
    WF += AmplVector[n] * np.exp(-1j * Eeig[n] * np.diag(T)) @ U[:, n]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histogram for the square of each amplitude
axs[0].bar(np.arange(Nstates), np.abs(AmplVector)**2, color='r')
axs[0].set_xlim(0, Nstates)

# Plot wave function
MaxVal = np.max(np.abs(WF)**2)
axs[1].plot(x, np.abs(WF)**2, 'k-', linewidth=2)
axs[1].set_xlim(-L/2, L/2)
axs[1].set_ylim(0, 1.5*MaxVal)
axs[1].axvline(-w/2, color='b', linestyle='--')
axs[1].axvline(w/2, color='b', linestyle='--')

plt.pause(0.1)

# Initiate time
t = 0
while t < Tmax:
    t += dt  # Update time
    # Calculate new wave function
    WF = np.zeros(N, dtype=complex)
    for n in range(Nstates):
        WF += AmplVector[n] * np.exp(-1j * Eeig[n] * np.diag(T)) @ U[:, n]
    # Update plot
    axs[1].plot(x, np.abs(WF)**2, 'k-', linewidth=2)
    axs[1].set_xlim(-L/2, L/2)
    axs[1].set_ylim(0, MaxVal)
    plt.draw()
    plt.pause(0.01)

# Draw a random number to emulate stochastic measurement outcome
Draw = np.random.rand()

# Accumulative probabilities
AccuProb = np.concatenate(([0], np.cumsum(np.abs(AmplVector)**2)))
Measure = np.where(Draw > AccuProb)[0].max()

# Update amplitude vector according to measurement outcome
AmplVector = np.zeros(Nstates)
AmplVector[Measure] = 1

# Collapse wave function
WF = U[:, Measure]

# Update plots
axs[0].bar(np.arange(Nstates), np.abs(AmplVector)**2, color='r')
axs[1].plot(x, np.abs(WF)**2, 'k-', linewidth=2)
axs[1].set_xlim(-L/2, L/2)
axs[1].set_ylim(0, np.max(np.abs(WF)**2))
plt.draw()
plt.show()
