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
% V     - (S x S) survival probability matrix (set 1 for infinitely-lived case)
%       if (1 x 1), then assume constant probability
% G     - (S^2 x J) matrix of asymptotic growth rates
%       if (S x J), then assume G does not depend on s'
%
% Optional:
% zetaBound     - (1 x 2) lower and upper bounds for searching for zeta
%                 default [1e-2,100]
%
% Output:
% zeta      - Pareto exponent
% typeDist  - (S x 1) probability distribution of types in upper tail
%             ([] when zeta could not be identified)
%
% Version 1.1: June 16, 2019
%
% Version 1.2: December 22, 2021
% - Allowed survival probability to be state-dependent
% - Added upper tail type distribution as output
%
% Version 1.3: August 1, 2026
% - BUG FIX: typeDist was left unassigned on the "no Pareto tail" early
%   return, which raised an error whenever two outputs were requested
% - Moved the shape checks on G ahead of the early returns so that a
%   misshapen G is always caught
% - Replaced eigs by the spectral radius max(abs(eig(.))) for the small
%   (S x S) matrix: eigs can fail or warn on tiny matrices, and the Perron
%   root is what the Beare-Toda formula calls for
% - Guarded against log(0) = -Inf when the matrix is nilpotent/zero
% - Validated that PS and PJ have unit row sums
%
% Version 1.4: August 16, 2026
% - Added explicit real-valued and finite-input validation
% - Treated a root exactly on a search bound as a valid root and computed
%   its upper-tail type distribution
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [zeta,typeDist] = getZeta(PS,PJ,V,G,zetaBound)
%% some error checking
if (nargin < 5)||isempty(zetaBound)
    zetaBound = [1e-2,100];
end

typeDist = []; % so that the second output is always defined

if (numel(zetaBound) ~= 2)||~isreal(zetaBound)||any(~isfinite(zetaBound(:)))||...
        (zetaBound(1) <= 0)||(zetaBound(1) >= zetaBound(2))
    error('zetaBound is invalid')
end
zetaLB = zetaBound(1);
zetaUB = zetaBound(2);

S = size(PS,2); % number of exogenous states
J = size(PJ,2); % number of transitory states

if size(PS,1) ~= S
    error('PS must be a square matrix')
end

if ~isreal(PS)||any(~isfinite(PS(:)))
    error('PS must contain only finite real values')
end

if any(PS(:) < 0)||any(abs(sum(PS,2) - 1) > 1e-8)
    error('PS must be a stochastic matrix (nonnegative, rows summing to 1)')
end

if ~isreal(PJ)||any(~isfinite(PJ(:)))
    error('PJ must contain only finite real values')
end

if any(PJ(:) < 0)||any(abs(sum(PJ,2) - 1) > 1e-8)
    error('rows of PJ must be nonnegative and sum to 1')
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

if ~isreal(V)||any(~isfinite(V(:)))
    error('V must contain only finite real values')
end

if any(V(:) < 0)||any(V(:) > 1)
    error('entries of V must be in [0,1]')
end

if ~isreal(G)||any(G(:) < 0)||any(~isfinite(G(:)))
    error('G must be nonnegative, finite, and real')
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

if max(G(:)) <= 1 % does not generate Pareto tail; just set to upper bound
    zeta = zetaUB;
    warning('model does not generate Pareto tails')
    return
end

%% use Beare & Toda (2017) formula to compute Pareto exponent

% A(z) is the (S x S) matrix whose spectral radius characterizes zeta;
% entry (s,t) is P(t|s)*V(s,t)*E[G^z | s,t]
Amat = @(z)(PS.*V.*(reshape(sum(PJ.*G.^z,2),S,S)'));

lambda = @(z)(logSpectralRadius(Amat(z))); % objective function

lambdaLB = lambda(zetaLB);
lambdaUB = lambda(zetaUB);
boundTol = 1e-12;

if lambdaLB > boundTol % function positive throughout the interval, hence no solution
    zeta = zetaLB; % set to lower bound
    warning('zeta is below lower bound')
    return
end

if lambdaUB < -boundTol % function negative throughout the interval, hence no solution
    zeta = zetaUB; % set to upper bound
    warning('zeta is above upper bound')
    return
end

if abs(lambdaLB) <= boundTol
    zeta = zetaLB;
elseif abs(lambdaUB) <= boundTol
    zeta = zetaUB;
else
    zeta = fzero(lambda,zetaBound);
end

%% upper tail type distribution: left Perron vector of A(zeta)
temp = Amat(zeta);
[Vec,D] = eig(temp'); % small matrix, full eig is cheap and robust
[~,imax] = max(real(diag(D)));
v = real(Vec(:,imax));
v = abs(v); % Perron vector is sign-definite; make it nonnegative
if sum(v) <= 0
    warning('could not normalize upper tail type distribution')
    typeDist = [];
else
    typeDist = v/sum(v);
end

end

%% helper: log of the spectral radius, with -Inf handled gracefully
function y = logSpectralRadius(A)
rho = max(abs(eig(A)));
if ~isfinite(rho)
    error('spectral radius is not finite; check PS, V, and G')
elseif rho <= 0
    y = -Inf;
else
    y = log(rho);
end
end
