%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getQ
% (c) 2019 Emilien Gouin-Bonenfant and Alexis Akira Toda
%
% Purpose:
%       Compute transition probability matrix using Pareto extrapolation
%
% Usage:
%       [Q,piStar] = getQ(PS,PJ,V,x0,xGrid,gstjn,Gstj,zeta)
%
% Inputs:
% PS    - (S x S) transition probability matrix of exogenous state
% PJ    - (S^2 x J) matrix of conditional probabilities of transitory state
%       if (1 x J), then assume distribution of j does not depend on (s,s')
%       if (S x J), then assume distribution of j depends only on s
% V     - (S x S) survival probability matrix (set 1 for infinitely-lived case)
%       if (1 x 1), then assume constant probability
% x0    - initial state variable of newborn agents (not used if V is all 1)
% xGrid - (1 x N) grid for state variable x (asset, wealth, etc.)
% gstjn - (S^2 x NJ) matrix of law of motion of x
%       if (S x NJ), then assume x does not depend on s'
%       column N*(j-1)+n holds the destination of grid point n in state j
%
% Optional:
% Gstj  - (S^2 x J) matrix of asymptotic slope of law of motion of x
%       if (S x J), then assume G does not depend on s'
%       pass [] to estimate it from the two largest grid points
% zeta  - Pareto exponent
%       pass [] or omit to compute it via getZeta
%
% Output:
% Q     - (SN x SN) SPARSE transition probability matrix of (s,x)
%       call full(Q) if a dense matrix is required
%
% Optional:
% piStar - (SN x 1) stationary distribution of Q (full column vector)
%
% Version 1.1: June 16, 2019
%
% Version 1.2: April 22, 2020
% - Fixed bug when Gstj is empty
%
% Version 1.3: December 22, 2021
% - Allowed survival probability to be state-dependent
% - Eliminated grid spacing for extrapolation from optional argument
%
% Version 1.4: August 1, 2026
% - BUG FIX: the birth/death guard read "if any(V) < 1", which compares a
%   logical row vector against 1 and is therefore always false whenever V
%   has nonzero columns. The x0 range check it guards never ran. Corrected
%   to "if any(V(:) < 1)"
% - BUG FIX: docstring Usage line listed the arguments as (...,V,xGrid,x0,...)
%   while the function signature is (...,V,x0,xGrid,...)
% - BUG FIX: the early return for zeta <= 1 left the second output
%   unassigned, raising an error when the caller asked for it
% - zeta is now also computed when it is passed as empty, matching how
%   Gstj is already handled
% - Replaced the unconditional, never-restored "warning off" with a saved
%   warning state restored via onCleanup
% - Stationary distribution: eigs(Q',1,1) applied shift-invert exactly at
%   the eigenvalue 1, i.e. factorized a singular matrix (the reason the
%   warnings were being suppressed). Now uses the largest-magnitude
%   eigenvalue, with a linear-system fallback, and forces the result to be
%   a genuine probability vector
% - Renamed second output pi -> piStar so it no longer shadows the builtin
% - Added NaN/Inf checks on gstjn and a sanity check on the row sums of Q
%
% Version 1.5: August 1, 2026 (performance)
% - Q is now assembled as a SPARSE matrix from (row,column,value) triplets
%   instead of a dense SN x SN array. Q has at most 2S(J+1) nonzeros per
%   row regardless of N, so the dense storage was O((SN)^2) for O(SN) worth
%   of information. This is the change that makes large N feasible
% - The interpolation index is now found by binary search (discretize)
%   applied to a whole block of destinations at once, replacing the
%   per-destination linear scan find(xGrid < x,1,'last'). The assembly cost
%   drops from O(S^2*J*N^2) to O(S^2*J*N*log N), and the innermost loop
%   over grid points is gone entirely
% - The survival probability V(s,t) is folded into the block weight, and
%   the reset block is written directly as triplets. Previously the code
%   formed bigV = kron(V,ones(N)) and Q0 = kron(PS,Qx0), two dense SN x SN
%   matrices, purely to apply a per-block scalar
% - The Pareto extrapolation mass is aggregated over the Nextra
%   hypothetical points with accumarray before being written out, since it
%   all lands in the single row N*s. Cost is now O(N) entries per block
%   rather than O(Nextra), and gstjnExtra is no longer materialized
% - The invalid-zeta early return no longer allocates a dense NaN(SN)
%   array, which was a potential out-of-memory failure at large N
%
% Version 1.6: August 16, 2026
% - Added explicit real-valued and finite-input validation
% - Checked the stationary-distribution residual before accepting the
%   eigenvector and after using the linear-system fallback
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Q,piStar] = getQ(PS,PJ,V,x0,xGrid,gstjn,Gstj,zeta)
%% some error checking

S = size(PS,2); % number of exogenous states
J = size(PJ,2); % number of transitory states
N = numel(xGrid); % number of wealth grid points

if ~isvector(xGrid)||N < 2||~isreal(xGrid)||any(~isfinite(xGrid(:)))
    error('xGrid must be a finite real vector with at least two elements')
end

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

if any(diff(xGrid) <= 0)
    error('xGrid must be in strictly increasing order')
end

xg = xGrid(:); % column copy, used for all indexing below
xMax = xg(N);
if xMax <= 0
    error('largest grid point must be positive')
end

if size(gstjn,2) ~= N*J
    error('size of PJ, xGrid, and gstjn must be consistent')
end

if ~isreal(gstjn)||any(~isfinite(gstjn(:)))
    error('gstjn must contain only finite real values')
end

hasReset = any(V(:) < 1);

if hasReset % there is birth/death
    if ~isscalar(x0)||~isreal(x0)||~isfinite(x0)
        error('x0 must be a finite real scalar when V contains an entry below 1')
    end
    if x0 < xg(1)
        error('it must be x0 >= min(xGrid)')
    elseif x0 >= xg(N)
        error('it must be x0 < max(xGrid)')
    end
end

if size(gstjn,1) == S % law of motion does not depend on next state
    gstjn = kron(gstjn,ones(S,1)); % replicate rows to make it S^2 x NJ
end
if size(gstjn,1) ~= S^2
    error('size of PS and gstjn must be consistent')
end

%% define optional arguments if not provided

indTop = N*(1:J); % columns of gstjn holding the largest grid point

if (nargin < 7)||isempty(Gstj) % asymptotic slope not provided
    Gstj = (gstjn(:,indTop) - gstjn(:,indTop-1))/(xg(N) - xg(N-1));
    % compute slope from two largest grid points
end

if size(Gstj,1) == S % law of motion does not depend on next state
    Gstj = kron(Gstj,ones(S,1)); % replicate rows to make it S^2 x J
end
if size(Gstj,1) ~= S^2
    error('size of PS and Gstj must be consistent')
end
if size(Gstj,2) ~= J
    error('size of PJ and Gstj must be consistent')
end

if ~isreal(Gstj)||any(~isfinite(Gstj(:)))||any(Gstj(:) <= 0)
    error('asymptotic slope must be positive, finite, and real')
end

if (nargin < 8)||isempty(zeta) % Pareto exponent not provided
    zeta = getZeta(PS,PJ,V,Gstj);
end

if ~isscalar(zeta)||~isreal(zeta)||~isfinite(zeta)
    error('zeta must be a finite real scalar')
end

if zeta <= 1
    warning('zeta must be larger than 1 for finite mean')
    Q = spdiags(NaN(S*N,1),0,S*N,S*N); % NaN signal without dense storage
    piStar = NaN(S*N,1);
    return
end

%% conditional probability on the hypothetical extra grid points

h = xg(N) - xg(N-1); % grid spacing for extrapolation
Nprime = N + max(max(max(ceil((xMax - gstjn(:,indTop))./(Gstj*h)),0)));
% number of grid points in hypothetical grid
Nextra = Nprime - N + 1;

if ~isfinite(Nextra)||(Nextra < 1)
    error('failed to determine the number of extrapolation points')
end
if Nextra > 1e7
    warning('Nextra = %d extrapolation points; consider a larger xMax or a coarser grid',Nextra)
end

r = ones(Nextra,1); % conditional probability on extra grid points
if Nextra > 1 % nothing to do if Nextra = 1
    temp = h/xMax;
    r(1:end-1) = zeta*temp*(1 + (0:Nextra-2)'*temp).^(-zeta-1); % Pareto density
    r(end) = (1 + (Nextra-1)*temp)^(-zeta); % Pareto tail probability
    r(end) = r(end) + zeta*temp*(1 + (Nextra-1)*temp)^(-zeta-1)/2; % adjustment for trapezoidal formula
    r = r/sum(r); % normalize to probability vector
end

%% assemble the transition probability matrix as sparse triplets

% Q has at most 2S(J+1) nonzeros per row whatever N is, so it is assembled
% from (row,column,value) triplets and handed to sparse(), which sums
% duplicate entries for us.

nInt = S^2*J*2*(N-1); % grid points 1..N-1, two nodes each
Ii = zeros(nInt,1); Ji = zeros(nInt,1); Vi = zeros(nInt,1);
pInt = 0;

nExt = S^2*J*N; % extrapolation mass, aggregated to at most N nodes
Ie = zeros(nExt,1); Je = zeros(nExt,1); Ve = zeros(nExt,1);
pExt = 0;

rowsInt = (1:N-1)';

for s = 1:S
    for t = 1:S
        st = S*(s-1)+t; % row of PJ, gstjn, Gstj for the pair (s,t)
        pst = PS(s,t)*V(s,t); % survival-weighted P(t | s)
        if pst == 0
            continue % nothing to contribute
        end
        for j = 1:J
            pstj = pst*PJ(st,j); % V(s,t)*P(j,t | s)
            if pstj == 0
                continue
            end

            % --- nonstochastic simulation on grid points 1..N-1 ---
            x = gstjn(st,N*(j-1)+(1:N-1)).'; % destinations
            [ind,theta] = locate(x,xg,N);
            idx = pInt + (1:2*(N-1));
            Ii(idx) = [N*(s-1)+rowsInt; N*(s-1)+rowsInt];
            Ji(idx) = [N*(t-1)+ind; N*(t-1)+ind+1];
            Vi(idx) = [(1-theta)*pstj; theta*pstj];
            pInt = pInt + 2*(N-1);

            % --- Pareto extrapolation from the largest grid point ---
            % every hypothetical point sends mass out of the same row N*s,
            % so aggregate over the Nextra points before writing them out
            xe = gstjn(st,N*j) + Gstj(st,j)*h*(0:Nextra-1)';
            [inde,thetae] = locate(xe,xg,N);
            w = accumarray([inde; inde+1],[r.*(1-thetae); r.*thetae],[N 1]);
            nz = find(w);
            idx = pExt + (1:numel(nz));
            Ie(idx) = N*s;
            Je(idx) = N*(t-1)+nz;
            Ve(idx) = w(nz)*pstj;
            pExt = pExt + numel(nz);
        end
    end
end

%% birth/death: newborns are placed at x0 regardless of where they came from

if hasReset
    [ind0,theta0] = locate(x0,xg,N);
    nRes = S^2*2*N;
    Ir = zeros(nRes,1); Jr = zeros(nRes,1); Vr = zeros(nRes,1);
    pRes = 0;
    rowsAll = (1:N)';
    for s = 1:S
        for t = 1:S
            c = (1-V(s,t))*PS(s,t); % probability of dying and moving to t
            if c == 0
                continue
            end
            idx = pRes + (1:2*N);
            Ir(idx) = [N*(s-1)+rowsAll; N*(s-1)+rowsAll];
            Jr(idx) = [repmat(N*(t-1)+ind0,N,1); repmat(N*(t-1)+ind0+1,N,1)];
            Vr(idx) = [repmat(c*(1-theta0),N,1); repmat(c*theta0,N,1)];
            pRes = pRes + 2*N;
        end
    end
else
    Ir = []; Jr = []; Vr = []; pRes = 0;
end

Q = sparse([Ii(1:pInt); Ie(1:pExt); Ir(1:pRes)], ...
           [Ji(1:pInt); Je(1:pExt); Jr(1:pRes)], ...
           [Vi(1:pInt); Ve(1:pExt); Vr(1:pRes)], S*N, S*N);

% sanity check: Q must be a stochastic matrix
rowErr = max(abs(sum(Q,2) - 1));
if rowErr > 1e-8
    warning('rows of Q deviate from 1 by up to %g; check the inputs',rowErr)
end

%% stationary distribution
if nargout > 1
    warnState = warning('off','all');
    restoreWarn = onCleanup(@()warning(warnState));
    piStar = [];
    try
        [v,~] = eigs(Q',1,'lm'); % unit eigenvalue is the largest in modulus
        v = abs(real(full(v)));
        if (sum(v) > 0)&&all(isfinite(v))
            piStar = v/sum(v);
            if norm(Q'*piStar-piStar,1) > 1e-8
                piStar = [];
            end
        end
    catch
        piStar = [];
    end
    if isempty(piStar) % fall back to solving piStar'*(Q-I) = 0 with sum = 1
        A = Q' - speye(S*N);
        A(S*N,:) = 1; % replace the redundant equation by the normalization
        b = sparse(S*N,1);
        b(S*N) = 1;
        piStar = full(A\b);
        piStar = max(piStar,0);
        if ~all(isfinite(piStar))||(sum(piStar) <= 0)
            error('failed to compute a finite stationary distribution')
        end
        piStar = piStar/sum(piStar);
    end
    statErr = norm(Q'*piStar-piStar,1);
    if statErr > 1e-8
        warning('stationary-distribution residual is %g',statErr)
    end
end

end

%% helper: locate destinations on the grid by binary search
% Returns, for each x, the index ind and weight theta such that mass
% (1-theta) goes to node ind and theta to node ind+1. Destinations at or
% below xGrid(1) go entirely to node 1; destinations at or above xGrid(N)
% go entirely to node N. Always 1 <= ind <= N-1, so ind+1 never exceeds N.
function [ind,theta] = locate(x,xg,N)

x = x(:);
ind = discretize(x,xg); % NaN outside [xg(1),xg(N)]
below = (x <= xg(1));
above = (x >= xg(N));
ind(below) = 1;
ind(above) = N-1;

if any(isnan(ind))
    error('destination of the law of motion is not a finite number')
end

theta = (x - xg(ind))./(xg(ind+1) - xg(ind));
theta(below) = 0;
theta(above) = 1;

end
