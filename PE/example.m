clear;
clc;
close all

set(0,'DefaultTextFontSize', 14)
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultLineLineWidth',2)

beta = 0.96; % discount factor including birth/death probability
p = 0.025; % birth/death probability
tau = 0.2;
PS = [1-tau tau; tau 1-tau]; % transition probability matrix
S = size(PS,1); % number of states
mu = [0.03 0.07]'; % expected log return in each state
sigma = 0.10; % volatility of returns
PJ = [1/2 1/2];
Gstj = beta*exp([mu-sigma mu+sigma]);
% matrix of gross return on wealth assuming EIS = 1
N = 100; % number of grid points
xMax = 10; % largest grid point
x0 = 1; % initial wealth
xGrid = linspace(0,xMax,N+1);
xGrid(1) = []; % wealth grid
gstjn = kron(Gstj,xGrid);

%% compute Pareto exponent
zetaBound = [0.1 10]; % bound to search for Pareto exponent
tic
zeta = getZeta(PS,PJ,p,Gstj,zetaBound);
toc

%% compute joint transition probability matrix
tic
[Q,pi] = getQ(PS,PJ,p,x0,xGrid,gstjn,Gstj,zeta);
toc

xDist = sum(reshape(pi,N,S),2); % wealth distribution

figure
plot(xGrid,xDist)
xlabel('Wealth')
ylabel('Probability mass')

%% compute top wealth shares
topProb = [0.001 0.01 0.1]; % top 0.1, 1, and 10%
tic
topShare = getTopShares(topProb,xGrid,xDist,zeta);
toc
