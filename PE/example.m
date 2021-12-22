clear;
clc;

%% figure formatting

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesTickLabelInterpreter','latex');
set(0,'DefaultLegendInterpreter', 'latex')
   
set(0,'DefaultTextFontSize', 14)
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultLineLineWidth',1)

temp = get(gca,'ColorOrder');
c1 = temp(1,:);
c2 = temp(2,:);
close all

beta = 0.96; % discount factor including birth/death probability
p = 0.04; % birth/death probability
V = 1-p; % survival probability
tau = 0.05; % transition probability
PS = [1-tau tau; tau 1-tau]; % transition probability matrix
S = size(PS,1); % number of states
mu = [0.03 0.07]'; % expected log return in each state
sigma = 0.10; % volatility of returns
PJ = [1/2 1/2];
Gstj = beta*exp([mu-sigma mu+sigma]);
% matrix of gross return on wealth assuming EIS = 1
N = 100; % number of grid points
xMin = 0; % lower endpoint
xMax = 1e4; % upper endpoint
x0 = 1; % initial wealth
xGrid = expGrid(xMin,xMax,x0,N);
gstjn = kron(Gstj,xGrid);

%% compute Pareto exponent
zetaBound = [0.1 10]; % bound to search for Pareto exponent
tic
[zeta,typeDist] = getZeta(PS,PJ,V,Gstj,zetaBound);
toc

%% compute joint transition probability matrix
tic
[Q,pi] = getQ(PS,PJ,V,x0,xGrid,gstjn,Gstj,zeta);
toc

xDist = sum(reshape(pi,N,S),2); % wealth distribution
xDistCDF = cumsum(xDist)';

figure
loglog(xGrid,1-xDistCDF)
xlabel('Wealth')
ylabel('Tail probability')

%% compute top wealth shares
topProb = [0.001 0.01 0.1]; % top 0.1, 1, and 10%
tic
topShare = getTopShares(topProb,xGrid,xDist,zeta);
toc
