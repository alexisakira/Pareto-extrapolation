%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% example
%
% Purpose:
%       Minimal end-to-end example of the Pareto extrapolation method of
%       Gouin-Bonenfant and Toda: compute the Pareto exponent, the joint
%       transition probability matrix, the stationary wealth distribution,
%       and top wealth shares.
%
% Version 1.1: August 1, 2026
% - Replaced get(gca,...) by get(groot,...) so that the script no longer
%   opens a stray figure just to read the default color order
% - Renamed the stationary distribution to piStar; the old name pi
%   shadowed the builtin constant pi for the rest of the session
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;
close all;

%% figure formatting

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesTickLabelInterpreter','latex');
set(0,'DefaultLegendInterpreter','latex')

set(0,'DefaultTextFontSize', 14)
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultLineLineWidth',1)

temp = get(groot,'defaultAxesColorOrder'); % does not create a figure
c1 = temp(1,:);
c2 = temp(2,:);

%% model parameters

beta = 0.96; % discount factor including birth/death probability
p = 0.04; % birth/death probability
V = 1-p; % survival probability
tau = 0.05; % transition probability
PS = [1-tau tau; tau 1-tau]; % transition probability matrix
S = size(PS,1); % number of states
mu = [0.03 0.07]'; % expected log return in each state
sigma = 0.10; % volatility of returns
PJ = [1/2 1/2]; % distribution of the transitory shock
Gstj = beta*exp([mu-sigma mu+sigma]);
% matrix of gross return on wealth assuming EIS = 1

%% grid

N = 100; % number of grid points
xMin = 0; % lower endpoint
xMax = 1e4; % upper endpoint
x0 = 1; % initial wealth
xGrid = expGrid(xMin,xMax,x0,N);
gstjn = kron(Gstj,xGrid); % law of motion, (S x NJ)

%% compute Pareto exponent

zetaBound = [0.1 10]; % bound to search for Pareto exponent
tic
[zeta,typeDist] = getZeta(PS,PJ,V,Gstj,zetaBound);
toc
fprintf('Pareto exponent zeta = %.4f\n',zeta);

%% compute joint transition probability matrix

tic
[Q,piStar] = getQ(PS,PJ,V,x0,xGrid,gstjn,Gstj,zeta);
toc

xDist = sum(reshape(piStar,N,S),2); % wealth distribution
xDistCDF = cumsum(xDist)';
xTail = 1-xDistCDF; % tail probability

% At the largest grid point the tail probability is zero in exact
% arithmetic, so cumsum roundoff leaves it at about -4e-16. loglog drops
% nonpositive data and issues "Negative data ignored", so plot only the
% points where the tail probability is genuinely positive.
ind = (xTail > 0);

figure
loglog(xGrid(ind),xTail(ind))
xlabel('Wealth')
ylabel('Tail probability')
grid on

%% compute top wealth shares

topProb = [0.001 0.01 0.1]; % top 0.1, 1, and 10%
tic
topShare = getTopShares(topProb,xGrid,xDist,zeta);
toc

for k = 1:length(topProb)
    fprintf('Top %g%% wealth share: %.4f\n',100*topProb(k),topShare(k));
end
