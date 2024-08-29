% This script plots the time evolution of a few bound states in
% a rectangular well. It fixes a few more or less random initial
% amplitudes and determines the dynamical linear combination of
% these few bound states according to the trivial phase evolution.
% Finally, an energy measurement is imposed, the wave functions is
% collapsed onto one of the energy eigenstates, which are stationary.
% 

% Inputs for the well
V0=-4;              % Deabth (negative)
w=10;               % Widths

% Additional inputs for numerical approach
L=40;               % Size of grid
N=1024;             % Number of gridpoints

% Time inputs
Tmax = 10;
dt = 0.025;

% Amplitudes (make sure that the number of nonzero amplitudes does not
% exceed the number of bound states.
AmplVector = [1 4 -2 3 0.5 2 1 .5 .2 .1];
AmplVector = AmplVector/sqrt(sum(abs(AmplVector).^2));     % Normalize
Nstates = length(AmplVector);

% Set up kinetic energy: FFT
x = linspace(-.5,.5,N+2)'*L;
x = x(1:N); 
h = x(2)-x(1);                                          % Spatial step size
wavenumFFT = 2*(pi/L)*1i*[(0:N/2-1), (-N/2:-1)]';       % Momentum vector, FFT
T=-0.5*ifft(diag(wavenumFFT.^2)*fft(eye(N)));
% Potential: Rectangular well
V=V0*(abs(x)<w/2);
% Hamiltonian
H=T+diag(V);

% Diagonalization
[U Eeig]=eig(H);
Eeig = diag(Eeig);
% Sort energies and states in ascending order
[Eeig I]=sort(Eeig);
U=U(:,I);
% Normalize
U = U/sqrt(h);
  
% Plott wave functions for bound states - along with histogram of
% probabilities
WF = zeros(1,N);
for n = 1:Nstates
  WF = WF + AmplVector(n)*U(:,n).';
end
figure(1)
subplot(1, 3, 1)
% Histogram for the square of each amplitude
plBar = bar(1:Nstates, abs(AmplVector).^2, 'r');
axis([0 Nstates 0 1])
subplot(1, 3, [2:3])
% Plot wave function
MaxVal = max(abs(WF).^2);
pl = plot(x,abs(WF).^2, 'k-', 'linewidth', 2);
axis([-L/2 L/2 0 1.5*MaxVal])
hold on
xline(-w/2,'b--')
xline(+w/2,'b--')
hold off
pause

% Initiate time
t = 0;
while t < Tmax
  t = t+dt;                 % Update time
  % Calculate new wave function
  WF = zeros(1,N);
  for n = 1:Nstates
    WF = WF + AmplVector(n)*exp(-1i*Eeig(n)*t)*U(:,n).';
  end
  % Update plot
  set(pl, 'ydata', abs(WF).^2)
  axis([-L/2 L/2 0 MaxVal])
  drawnow
end

% Draw a random number to emulate stochastic measurement outcome
Draw = rand;
% Accumulative probabilities
AccuProb = [0, cumsum(abs(AmplVector).^2)];      
Measure = max(find(Draw > AccuProb));
% Update amplitude vector according to measurement outcome
AmplVector = zeros(1, Nstates);
AmplVector(Measure) = 1;
% Collapse wave function 
WF = U(:,Measure);
% Update plots
set(plBar, 'ydata', abs(AmplVector).^2)
set(pl, 'ydata', abs(WF).^2)
drawnow