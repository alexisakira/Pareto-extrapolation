%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getTopShares
% (c) 2019 Emilien Gouin-Bonenfant and Alexis Akira Toda
%
% Purpose:
%       Compute top wealth shares from wealth grid and stationary
%       distribution
%
% Usage:
%       topShare = getTopShares(topProb,wGrid,wDist,zeta)
%
% Inputs:
% topProb   - top probabilities to evaluate top shares. For example,
%           topProb = [0.001 0.01 0.1] evaluates top 0.1%, 1%, 10% shares
%           must be strictly increasing, with entries in [0,1]
% wGrid     - wealth grid (strictly increasing)
% wDist     - wealth distribution on wGrid (probability vector)
% zeta      - Pareto exponent (set 0 if using truncation)
%
% Outputs:
% topShare  - top wealth share corresponding to topProb, returned with the
%             same orientation (row/column) as topProb. If wGrid contains
%             negative wealth, topShare may exceed 1 and need not be
%             nondecreasing in topProb
%
% Version 1.1: June 16, 2019
%
% Version 1.2: April 20, 2020
% - Fixed bug in spline interpolation when the grid is fine
%
% Version 1.3: August 1, 2026
% - Switched interpolation from 'spline' to 'pchip' throughout, and removed
%   the optional method argument. A top share curve is monotone and
%   bounded; 'spline' is not shape preserving and can overshoot or dip
%   below zero near the flat part of the curve, whereas 'pchip' cannot
% - Output orientation now matches the orientation of topProb (previously
%   a row input returned a column output)
% - Added validation of zeta and of the row sum of wDist
% - The output is now explicitly enforced to be nonnegative, no greater
%   than 1, and nondecreasing in topProb. With 'pchip' on nonnegative
%   monotone data these bounds are already satisfied, so this is a
%   safeguard rather than a correction
%
% Version 1.4: August 16, 2026
% - Allowed negative wealth. Top wealth shares can legitimately exceed 1
%   and can decline as indebted households enter the top group, so removed
%   the [0,1] clamp and monotonicity enforcement introduced in Version 1.3
% - Required aggregate net wealth (including the Pareto correction, when
%   used) to be strictly positive
% - Added explicit vector, real-valued, and finite-input validation
% - Normalized wDist after checking that it sums to one within tolerance
% - Handled zero probability at the largest grid point without creating
%   duplicate interpolation knots at topProb = 0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function topShare = getTopShares(topProb,wGrid,wDist,zeta)
%% some error checking

if (nargin < 4)||isempty(zeta)
    zeta = 0; % use truncation
end

if ~isscalar(zeta)||~isreal(zeta)||~isfinite(zeta)||(zeta < 0)
    error('zeta must be a nonnegative finite scalar (0 to use truncation)')
end

if ~(isvector(topProb)||isempty(topProb))||~isreal(topProb)||any(~isfinite(topProb(:)))
    error('topProb must be a finite real vector')
end

if any(topProb(:) < 0)||any(topProb(:) > 1)
    error('topProb must be vector of numbers between 0 and 1')
end

if any(diff(topProb(:)) <= 0)
    error('topProb must be strictly increasing')
end

if ~isvector(wGrid)||isempty(wGrid)||~isreal(wGrid)||any(~isfinite(wGrid(:)))
    error('wGrid must be a nonempty finite real vector')
end

if ~isvector(wDist)||isempty(wDist)||~isreal(wDist)||any(~isfinite(wDist(:)))
    error('wDist must be a nonempty finite real vector')
end

if any(diff(wGrid(:)) <= 0)
    error('wGrid must be strictly increasing')
end

if numel(wGrid) ~= numel(wDist)
    error('length of wGrid and wDist must agree')
end

if any(wDist(:) < 0)
    error('wDist must be nonnegative')
end

distSum = sum(wDist(:));
if abs(distSum - 1) > 1e-6
    error('wDist must sum to 1 (its sum is %g)',distSum)
end

isRowInput = isrow(topProb); % remember orientation

wGrid = wGrid(:);
wDist = wDist(:)/distSum; % remove harmless probability roundoff
topProb = topProb(:);

tailProb = cumsum(flipud(wDist)); % tail probability
[~,ia,~] = unique(tailProb); % index of unique values

%% first, consider when using truncation
if zeta == 0
    aggW = dot(wDist,wGrid); % aggregate wealth
    if aggW <= 0
        error('aggregate net wealth must be strictly positive')
    end
    topWealth = cumsum(flipud(wDist.*wGrid))/aggW; % top wealth shares on grid
    [probKnots,ib] = unique([0;tailProb(ia)]);
    wealthKnots = [0;topWealth(ia)];
    topShare = safeInterp1(probKnots,wealthKnots(ib),topProb);
    topShare = restoreOrientation(topShare,isRowInput);
    return
end

%% next, consider Pareto extrapolation
if zeta <= 1
    error('zeta must exceed 1') % need zeta > 1 for finite mean
end
if wGrid(end) <= 0
    error('largest wealth grid point must be positive for Pareto extrapolation')
end

temp = wGrid;
temp(end) = (zeta/(zeta-1))*wGrid(end); % correct last grid point
aggW = dot(wDist,temp); % aggregate wealth using Pareto extrapolation
if aggW <= 0
    error('aggregate net wealth including the Pareto correction must be strictly positive')
end
topWealth = cumsum(flipud(wDist.*temp))/aggW; % top wealth shares on grid

ind1 = find(topProb <= tailProb(1)); % index for which extrapolation is necessary
ind2 = find(topProb > tailProb(1)); % index for which extrapolation is unnecessary

topShare = 0*topProb;
topShare(ind1) = (zeta/(zeta-1))*wDist(end)^(1/zeta)*(wGrid(end)/aggW)*topProb(ind1).^(1-1/zeta);
% extrapolate top wealth shares using Pareto distribution
topShare(ind2) = safeInterp1(tailProb(ia),topWealth(ia),topProb(ind2));

topShare = restoreOrientation(topShare,isRowInput);

end

%% helper: shape-preserving interpolation that never extrapolates
function yq = safeInterp1(x,y,xq)

if isempty(xq)
    yq = xq;
    return
end

nPts = numel(x);
if nPts < 2
    error('not enough distinct points to interpolate top shares')
end

if nPts >= 3
    method = 'pchip'; % shape preserving: cannot overshoot or undershoot
else
    method = 'linear'; % pchip needs at least 3 points
end

xq = min(max(xq,min(x)),max(x)); % clamp: never extrapolate
yq = interp1(x,y,xq,method);

end

%% helper: validate the result and restore input orientation
function y = restoreOrientation(y,isRowInput)

if any(~isfinite(y))
    error('failed to compute finite top wealth shares; check the inputs')
end

if isRowInput
    y = y.';
end

end
