 function CAVE = cave(U, X, varargin)
%==========================================================================
% This function computes the cumlative amount of variance explained (CAVE)
% by up to k component vectors from a tensor decomposition. These component
% vectors may result from either rho-PCA or CP-decomposition. Refer to
% Theorem 2 from [1] for a more thorough explaination.
%
% INPUT:
%   1. 'U': (4 x 1) cell array of (n x K) factors, where n is the
%      dimensionaltiy of the respective mode. This is output from rhopca.m
%   2. 'X': The original tensor
%   3. 'k': The number of factors to include in calculation.
%   4. 'verbose' (default true): Optional, boolean value indicating to 
%       print all messages in code
%
% OUTPUT:
%   1. 'CAVE': Amount of variance explained, vector if k > 1.
%
% Dependencies: Tensor Toolbox package
%
% References
% 1. Allen, Genevera. "Sparse higher-order principal components analysis."
% Artificial Intelligence and Statistics. 2012.
%==========================================================================

%% Load and verify parameters
% Specify default parameters and their expected values
% default_verbose = true;
default_k = 1;
% Parse input parameters
p = inputParser;
p.CaseSensitive = true;
addRequired(p, 'U');
addRequired(p, 'X', @(ii) isa(ii,'double')||isa(ii,'tensor'));
addParameter(p, 'k', default_k, @(ii) (isnumeric(ii)&&(ii > 0)))
parse(p, U, X, varargin{:});
k = p.Results.k;
% Check dimensionality of covariates X
X = squeeze(X);         % Remove singletion dimensions, if present
% Force all values to be positive
X = double(X);
% ii = X < 0;
% X(ii) = abs(X(ii));
% Convert X to tensor class
if ~isa(X, 'tensor'); X = tensor(X); end
% Check to see if the number of factors is the same for each mode
nModes = length(U.U);
nFactors = size(U{1},2);
if nFactors < 1
    error('There must be at least one factor for each mode.'); 
end
for mm = 2:nModes
    if nFactors ~= size(U{mm},2)
        error('The number of factors is not the same for all modes.\n');
    end
end
% Check k vs the number of factors
if nFactors < k
    error('The value k cannot be greater than the number of factors.\n');
end

%% Calcualte the cumlative amount of variance explained for each factor
CAVE = zeros(k,1);
for jj = 1:k
    % Calculate the project matrices
    P = cell(nModes,1);
    for mm = 1:nModes
        P{mm} = U{mm}(:,1:jj)*inv(transpose(U{mm}(:,1:jj))*U{mm}(:,1:jj))*transpose(U{mm}(:,1:jj));
    end
    % Calculate the PCs
    PC = ttm(ttm(ttm(ttm(X, P{1}, 1), P{2}, 2), P{3}, 3), P{4}, 4);
    numer = norm(tensor(PC))^2;
    denom = norm(tensor(X))^2;
    CAVE(jj) = numer/denom;
end
end
