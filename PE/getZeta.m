%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getZeta
% (c) 2019 Emilien Gouin-Bonenfant and Alexis Akira Toda
% 
% Purpose: 
%       Compute Pareto exponent of Markov multiplicative process with reset
%       using Beare & Toda (2017) formula
%
% Usage:
%       [zeta,typeDist] = getZeta(PS,PJ,V,G,zetaBound)
%
% Inputs:
% PS    - (S x S) transition probability matrix of exogenous state
% PJ    - (S^2 x J) matrix of conditional probabilities of transitory state
%       if (1 x J), then assume distribution of j does not depend on (s,s')
%       if (S x J), then assume distribution of j depends only on s
% V     - (S x S) survival probability matrix (set 0 for infinitely-lived case)
%       if (1 x 1), then assume constant probability
% G     - (S^2 x J) matrix of asymptotic growth rates
%       if (S x J), then assume G does not depend on s'
%
% Optional:
% zetaBound     - lower and upper bounds for searching for zeta
%
% Output:
% zeta      - Pareto exponent
% typeDist  - probability distribution of types in upper tail
%
% Version 1.1: June 16, 2019
%
% Version 1.2: December 22, 2021
% - Allowed survival probability to be state-dependent
% - Added upper tail type distribution as output
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [zeta,typeDist] = getZeta(PS,PJ,V,G,zetaBound)
%% some error checking
if nargin < 5
    zetaBound = [1e-2,100];
end

if (length(zetaBound) ~= 2)||(zetaBound(1) <= 0)||(zetaBound(1) >= zetaBound(2))
    error('zetaBound is invalid')
end
zetaLB = zetaBound(1);
zetaUB = zetaBound(2);

S = size(PS,2); % number of exogenous states
J = size(PJ,2); % number of transitory states

if size(PS,1) ~= S
    error('PS must be a square matrix')
end

if size(PJ,1) == 1 % conditional distribution independent of states
    PJ = repmat(PJ,S^2,1);
elseif size(PJ,1) == S % conditional distribution depends only current state
    PJ = kron(PJ,ones(S,1));
elseif size(PJ,1) ~= S^2
    error('size of PS and PJ inconsistent')
end

if isscalar(V)
    V = V*ones(S);
end
if (size(V,1) ~= S)||(size(V,2) ~= S)
    error('size of PS and V inconsistent')
end

if any(V(:) < 0)||any(V(:) > 1)
    error('entries of V must be in [0,1]')
end

if any(G(:) < 0)
    error('G must be nonnegative')
end

if max(G(:)) <= 1 % does not generate Pareto tail; just set to upper bound
    zeta = zetaUB;
    warning('model does not generate Pareto tails')
    return
end

if size(G,1) == S % law of motion does not depend on next state
    G = kron(G,ones(S,1)); % replicate rows to make it S^2 x J
end

if size(G,1) ~= S^2
    error('size of PS and G inconsistent')
end

if size(G,2) ~= J
    error('size of PJ and G inconsistent')
end

%% use Beare & Toda (2017) formula to compute Pareto exponent
lambda = @(z)(log(eigs(PS.*V.*(reshape(sum(PJ.*G.^z,2),S,S)'),1))); % objective function

if lambda(zetaLB) >= 0 % function positive for all zeta in [zetaLB,zetaUB], hence no solution
    zeta = zetaLB; % set to lower bound
    warning('zeta is below lower bound')
    typeDist = [];
    return
end

if lambda(zetaUB) <= 0 % function negative for all zeta in [zetaLB,zetaUB], hence no solution
    zeta = zetaUB; % set to upper bound
    warning('zeta is above upper bound')
    typeDist = [];
    return
end

zeta = fzero(lambda,zetaBound);
temp = PS.*V.*(reshape(sum(PJ.*G.^zeta,2),S,S)');
[v,~] = eigs(temp',1,1);
typeDist = v/sum(v);

end
