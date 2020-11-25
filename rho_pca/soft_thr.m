function u = soft_thr(a, lam, varargin)
%==========================================================================
% Soft thresholding function
% 
% INPUT:
%   1. a - vector
%   2. lam - thesholding parameter
%   3. pos - boolean value indicating if value is non-negative, default is
%   false
%
% OUTPUT:
%   1. u - sparse vector
%
% Ref: http://eeweb.poly.edu/iselesni/lecture_notes/SoftThresholding.pdf
%==========================================================================

%% Load and verify parameters
default_pos = false;
p = inputParser;
p.CaseSensitive = true;
addRequired(p, 'a', @(ii) isa(ii,'double'));
addRequired(p, 'lam', @(ii) (isnumeric(ii)&&(ii >= 0)));
addParameter(p, 'pos', default_pos, @islogical);
parse(p, a, lam, varargin{:});
a = p.Results.a;
lam = p.Results.lam;
pos = p.Results.pos;

%% Perform soft thesholding
if pos
    u = max(a - lam, 0);
else
    u = sign(a).*max(abs(a) - lam, 0);
end

end



