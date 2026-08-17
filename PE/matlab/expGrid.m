%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% expGrid
% (c) 2019 Alexis Akira Toda
%
% Purpose:
%       Construct an exponential grid
%
% Usage:
%       grid = expGrid(a,b,c,N)
%
% Inputs:
% a - lower endpoint of grid (this IS the first grid point)
% b - upper endpoint of grid (this IS the last grid point)
% c - median grid point
% N - number of grid points (integer >= 2)
%
% Output:
% grid  - (1 x N) strictly increasing row vector with grid(1) = a and
%         grid(N) = b
%
% Version 1.1: June 16, 2019
%
% Version 1.2: August 3, 2026
% - Validated N and the shift parameter s
% - Pinned grid(1) and grid(end) to a, b exactly (removes roundoff)
%
% Version 1.3: August 16, 2026
% - Required all inputs, including N, to be finite and real
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grid = expGrid(a,b,c,N)

% some error checking
if ~isscalar(a)||~isscalar(b)||~isscalar(c)||~isscalar(N)
    error('a, b, c, N must all be scalars')
end

if ~isreal(a)||~isreal(b)||~isreal(c)||~isreal(N)||...
        ~isfinite(a)||~isfinite(b)||~isfinite(c)||~isfinite(N)
    error('a, b, c, N must be finite and real')
end

if (a >= b)||(c <= a)||(c >= (a+b)/2)
    error('it must be a < c < (a+b)/2')
end

if (N < 2)||(N ~= floor(N))
    error('N must be an integer no less than 2')
end

s = (c^2-a*b)/(a+b-2*c); % shift parameter

if a+s <= 0
    error('shift parameter too small: need a+s > 0 (try a larger c)')
end

temp = linspace(log(a+s),log(b+s),N); % even grid in log scale
grid = exp(temp)-s;
grid(1) = a; % enforce lower endpoint exactly
grid(end) = b; % enforce upper endpoint exactly

end
